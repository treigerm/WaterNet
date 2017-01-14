import fiona
import pickle
import rasterio
import rasterio.features
import rasterio.warp
import os
from config import TILES_DIR
from config import WATER_BITMAPS_DIR
from geo_util import create_tiles
from geo_util import read_bands
from geo_util import read_geotiff
from geo_util import reproject_dataset
from io_util import get_file_name, save_tiles, save_tiles, save_image
import numpy as np


def preprocess_data(tile_size, dataset, only_cache=False):
    print('_' * 100)
    print("Start preprocessing data.")

    features_train, labels_train = extract_features_and_labels(
        dataset["train"], tile_size, only_cache)
    features_test, labels_test = extract_features_and_labels(
        dataset["test"], tile_size, only_cache)

    return features_train, features_test, labels_train, labels_test


def extract_features_and_labels(dataset, tile_size, only_cache=False):
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

    # TODO: Comments
    dataset, wgs84_path = reproject_dataset(geotiff_path)
    bands = read_bands(dataset)

    # For the given satellite image create a bitmap which has 1 at every pixel which corresponds
    # to water in the satellite image. In order to do this we use water polygons from OpenStreetMap.
    # The water polygons are stored in forms of shapefiles and are given by "shapefile_paths".
    water_bitmap = create_bitmap(dataset, shapefile_paths, geotiff_path)

    # Tile the RGB bands of the satellite image and the bitmap.
    tiled_bands = create_tiles(bands, tile_size, wgs84_path)
    tiled_bitmap = create_tiles(water_bitmap, tile_size, wgs84_path)

    tiled_bands, tiled_bitmap = remove_edge_tiles(tiled_bands, tiled_bitmap,
                                                  tile_size, dataset.shape)

    save_tiles(cache_path, tiled_bands, tiled_bitmap)

    return tiled_bands, tiled_bitmap


def remove_edge_tiles(tiled_bands, tiled_bitmap, tile_size, source_shape):
    EDGE_BUFFER = 350
    rows, cols = source_shape[0], source_shape[1]

    bands = []
    bitmap = []
    for i, (tile, (row, col), _) in enumerate(tiled_bands):
        is_in_center = EDGE_BUFFER <= row and row <= (
            rows - EDGE_BUFFER) and EDGE_BUFFER <= col and col <= (
                cols - EDGE_BUFFER)
        # Checks wether our tile contains a pixel which is only black.
        contains_black_pixel = [0, 0, 0] in tile
        is_edge_tile = contains_black_pixel and not is_in_center
        if not is_edge_tile:
            bands.append(tiled_bands[i])
            bitmap.append(tiled_bitmap[i])

    return bands, bitmap


def create_bitmap(raster_dataset, shapefile_paths, satellite_path):
    satellite_img_name = get_file_name(satellite_path)
    cache_file_name = "{}_water.tif".format(satellite_img_name)
    cache_path = os.path.join(WATER_BITMAPS_DIR, cache_file_name)
    try:
        print("Load water bitmap from {}".format(cache_path))
        _, image = read_geotiff(cache_path)
        image[image == 255] = 1
        return image
    except IOError as e:
        print("No cache file found.")

    water_features = np.empty((0, ))

    print("Create bitmap for water features.")
    for shapefile_path in shapefile_paths:
        print("Load shapefile {}.".format(shapefile_path))
        with fiona.open(shapefile_path) as shapefile:
            # Each feature in the shapefile also contains meta information such as
            # wether the features is a lake or a river. We only care about the geometry
            # of the feature i.e. where it is located and what shape it has.
            geometries = [feature['geometry'] for feature in shapefile]

            water_features = np.concatenate(
                (water_features, geometries), axis=0)

    # Now that we have the vector data of all water features in our satellite image
    # we "burn it" into a new raster so that we get a B/W image with water features
    # in white and the rest in black.
    bitmap_image = rasterio.features.rasterize(
        ((g, 255) for g in water_features),
        out_shape=raster_dataset.shape,
        transform=raster_dataset.transform)

    save_image(cache_path, bitmap_image, raster_dataset)

    bitmap_image[bitmap_image == 255] = 1
    return bitmap_image
