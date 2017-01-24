"""Transform GeoTIFF images and OSM shapefiles into feature and label matrixes which can be use
to train the ConvNet"""

import fiona
import pickle
import rasterio
import rasterio.features
import rasterio.warp
import os
import sys
from config import TILES_DIR, WATER_BITMAPS_DIR
from geo_util import create_tiles, reproject_dataset
from io_util import get_file_name, save_tiles, save_tiles, save_bitmap, load_bitmap
import numpy as np


def preprocess_data(tile_size, dataset, only_cache=False):
    """Create features and labels for a given dataset. The features are tiles which contain
    the three RGB bands of the satellite image, so they have the form (tile_size, tile_size, 3).
    Labels are bitmaps with 1 indicating that the corresponding pixel in the satellite image
    represents water."""

    print('_' * 100)
    print("Start preprocessing data.")

    features_train, labels_train = extract_features_and_labels(
        dataset["train"], tile_size, only_cache)
    features_test, labels_test = extract_features_and_labels(
        dataset["test"], tile_size, only_cache)

    return features_train, features_test, labels_train, labels_test


def extract_features_and_labels(dataset, tile_size, only_cache=False):
    """For each satellite image and its corresponding shapefiles in the dataset create
    tiled features and labels."""
    features = []
    labels = []

    for geotiff_path, shapefile_paths in dataset:
        tiled_features, tiled_labels = create_tiled_features_and_labels(
            geotiff_path, shapefile_paths, tile_size, only_cache)

        features += tiled_features
        labels += tiled_labels

    return features, labels


def create_tiled_features_and_labels(geotiff_path,
                                     shapefile_paths,
                                     tile_size,
                                     only_cache=False):
    """Create the features and labels for a given satellite image and its shapefiles."""

    # Try to load tiles from cache.
    satellite_img_name = get_file_name(geotiff_path)
    cache_file_name = "{}_{}.pickle".format(satellite_img_name, tile_size)
    cache_path = os.path.join(TILES_DIR, cache_file_name)
    try:
        print("Load tiles from {}.".format(cache_path))
        with open(cache_path) as f:
            tiles = pickle.load(f)

        return tiles["features"], tiles["labels"]
    except IOError as e:
        if only_cache:
            raise
        print("Cache not available. Compute tiles.")

    # The provided satellite images have a different coordinate reference system as
    # the familiar WGS 84 which uses Latitude and Longitude. So we need to reproject
    # the satellite image to the WGS 84 coordinate reference system.
    dataset, wgs84_path = reproject_dataset(geotiff_path)
    bands = np.dstack(dataset.read())

    # For the given satellite image create a bitmap which has 1 at every pixel which corresponds
    # to water in the satellite image. In order to do this we use water polygons from OpenStreetMap.
    # The water polygons are stored in forms of shapefiles and are given by "shapefile_paths".
    water_bitmap = create_bitmap(dataset, shapefile_paths, geotiff_path)

    # Tile the RGB bands of the satellite image and the bitmap.
    tiled_bands = create_tiles(bands, tile_size, wgs84_path)
    tiled_bitmap = create_tiles(water_bitmap, tile_size, wgs84_path)

    # Due to the projection the satellite image in the GeoTIFF is not a perfect rectangle and the
    # remaining space on the edges is blacked out. When we overlay the GeoTIFF with the
    # shapefile it also overlays features for the blacked out parts which means that if we don't
    # remove these tiles the classifier will be fed with non-empty labels for empty features.
    tiled_bands, tiled_bitmap = remove_edge_tiles(tiled_bands, tiled_bitmap,
                                                  tile_size, dataset.shape)

    save_tiles(cache_path, tiled_bands, tiled_bitmap)

    return tiled_bands, tiled_bitmap


def remove_edge_tiles(tiled_bands, tiled_bitmap, tile_size, source_shape):
    """Remove tiles which are on the edge of the satellite image and which contain blacked out
    content."""

    EDGE_BUFFER = 350

    rows, cols = source_shape[0], source_shape[1]

    bands = []
    bitmap = []
    for i, (tile, (row, col), _) in enumerate(tiled_bands):
        is_in_center = EDGE_BUFFER <= row and row <= (
            rows - EDGE_BUFFER) and EDGE_BUFFER <= col and col <= (
                cols - EDGE_BUFFER)
        # Checks wether our tile contains a pixel which is only black.
        # This might also delete tiles which contain a natural feature which is
        # totally black but these are only a small number of tiles and we don't
        # care about deleting them as well.
        contains_black_pixel = [0, 0, 0] in tile
        is_edge_tile = contains_black_pixel and not is_in_center
        if not is_edge_tile:
            bands.append(tiled_bands[i])
            bitmap.append(tiled_bitmap[i])

    return bands, bitmap


def create_bitmap(raster_dataset, shapefile_paths, satellite_path):
    """Create the bitmap for a given satellite image."""

    satellite_img_name = get_file_name(satellite_path)
    cache_file_name = "{}_water.tif".format(satellite_img_name)
    cache_path = os.path.join(WATER_BITMAPS_DIR, cache_file_name)
    try:
        # Try loading the water bitmap from cache.
        print("Load water bitmap from {}".format(cache_path))
        bitmap = load_bitmap(cache_path)
        bitmap[bitmap == 255] = 1
        return bitmap
    except IOError as e:
        print("No cache file found.")

    water_features = np.empty((0, ))

    print("Create bitmap for water features.")
    for shapefile_path in shapefile_paths:
        try:
            print("Load shapefile {}.".format(shapefile_path))
            with fiona.open(shapefile_path) as shapefile:
                # Each feature in the shapefile also contains meta information such as
                # wether the features is a lake or a river. We only care about the geometry
                # of the feature i.e. where it is located and what shape it has.
                geometries = [feature['geometry'] for feature in shapefile]

                water_features = np.concatenate(
                    (water_features, geometries), axis=0)
        except IOError as e:
            print("No shapefile found.")
            sys.exit(1)

    # Now that we have the vector data of all water features in our satellite image
    # we "burn it" into a new raster so that we get a B/W image with water features
    # in white and the rest in black. We choose the value 255 so that there is a stark
    # contrast between water and non-water pixels. This is only for visualisation
    # purposes. For the classifier we use 0s and 1s.
    bitmap_image = rasterio.features.rasterize(
        ((g, 255) for g in water_features),
        out_shape=raster_dataset.shape,
        transform=raster_dataset.transform)

    save_bitmap(cache_path, bitmap_image, raster_dataset)

    bitmap_image[bitmap_image == 255] = 1
    return bitmap_image
