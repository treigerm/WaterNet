import fiona
import itertools
import pickle
import rasterio
import rasterio.features
import rasterio.warp
import time
import os
from config import TILES_DIR, WATER_POLYGONS_DIR, WATER_BITMAPS_DIR, SHAPEFILE_CHECKPOINTS_DIR, WGS84_DIR
from coordinate_translation import lat_lon_to_raster_crs
from process_geotiff import read_geotiff, read_bands, read_bitmap, create_tiles, image_from_tiles, overlay_bitmap
import numpy as np


def preprocess_data(tile_size, dataset):

    print('_' * 100)
    print("Start preprocessing data.")

    features_train, labels_train = extract_features_and_labels(
        dataset["train"], tile_size)
    features_test, labels_test = extract_features_and_labels(
        dataset["test"], tile_size)

    return features_train, features_test, labels_train, labels_test


def extract_features_and_labels(dataset, tile_size):
    features = []
    labels = []

    for geotiff_path, shapefile_paths in dataset:
        tiled_features, tiled_labels = create_tiled_features_and_labels(
            geotiff_path, shapefile_paths, tile_size)

        features += tiled_features
        labels += tiled_labels

    return features, labels


def create_tiled_features_and_labels(geotiff_path, shapefile_paths, tile_size):
    # Try to load tiles from cache.
    satellite_img_name = get_file_name(geotiff_path)
    cache_file_name = "{}_{}.pickle".format(
        satellite_img_name, tile_size)
    cache_path = os.path.join(TILES_DIR, cache_file_name)
    try:
        print("Load tiles from {}.".format(cache_path))
        with open(cache_path) as f:
            tiles = pickle.load(f)

        return tiles["features"], tiles["labels"]
    except IOError as e:
        print("Cache not available. Compute tiles.")

    # TODO: Comments
    dataset = reproject_dataset(geotiff_path)
    bands = read_bands(dataset)

    # For the given satellite image create a bitmap which has 1 at every pixel which corresponds
    # to water in the satellite image. In order to do this we use water polygons from OpenStreetMap.
    # The water polygons are stored in forms of shapefiles and are given by "shapefile_paths".
    water_bitmap = create_bitmap(dataset, shapefile_paths, geotiff_path)

    # Tile the RGB bands of the satellite image and the bitmap.
    tiled_bands = create_tiles(bands, tile_size, geotiff_path)
    tiled_bitmap = create_tiles(water_bitmap, tile_size, geotiff_path)

    tiled_bands, tiled_bitmap = remove_edge_tiles(tiled_bands, tiled_bitmap, tile_size, dataset.shape)

    save_tiles(cache_path, tiled_bands, tiled_bitmap)

    return tiled_bands, tiled_bitmap

def remove_edge_tiles(tiled_bands, tiled_bitmap, tile_size, source_shape):
    EDGE_BUFFER = 350
    rows, cols = source_shape[0], source_shape[1]

    bands = []
    bitmap = []
    for i, (tile, (row, col), _) in enumerate(tiled_bands):
        is_in_center = EDGE_BUFFER <= row and row <= (rows - EDGE_BUFFER) and EDGE_BUFFER <= col and col <= (cols - EDGE_BUFFER)
        # Checks wether our tile contains a pixel which is only black. 
        contains_black_pixel = [0,0,0] in tile
        is_edge_tile = contains_black_pixel and not is_in_center
        if not is_edge_tile:
            bands.append(tiled_bands[i])
            bitmap.append(tiled_bitmap[i])

    return bands, bitmap

def reproject_dataset(geotiff_path):
    dst_crs = 'EPSG:4326'

    with rasterio.open(geotiff_path) as src:
        transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        satellite_img_name = get_file_name(geotiff_path)
        out_file_name = "{}_wgs84.tif".format(satellite_img_name)
        out_path = os.path.join(WGS84_DIR, out_file_name)
        with rasterio.open(out_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=rasterio.warp.Resampling.nearest)

        return rasterio.open(out_path)


def create_bitmap(raster_dataset, shapefile_paths, satellite_path):
    satellite_img_name = get_file_name(satellite_path)
    cache_file_name = "{}_water.tif".format(
        satellite_img_name)
    cache_path = os.path.join(WATER_BITMAPS_DIR, cache_file_name)
    try:
        print("Load water bitmap from {}".format(cache_path))
        _, image = read_geotiff(cache_path)
        # TODO: Don't resize image but change create_tiles.
        image = np.reshape(image, (image.shape[0], image.shape[1], 1))
        image[image == 255] = 1
        return image
    except IOError as e:
        print("No cache file found.")

    water_features = np.empty((0,))

    print("Create bitmap for water features.")
    for shapefile_path in shapefile_paths:
        print("Load shapefile {}.".format(shapefile_path))
        with fiona.open(shapefile_path) as shapefile:
            # Each feature in the shapefile also contains meta information such as
            # wether the features is a lake or a river. We only care about the geometry
            # of the feature i.e. where it is located and what shape it has.
            geometries = [feature['geometry'] for feature in shapefile]

            water_features = np.concatenate((water_features, geometries), axis=0)

    # Now that we have the vector data of all water features in our satellite image
    # we "burn it" into a new raster so that we get a B/W image with water features
    # in white and the rest in black.
    bitmap_image = rasterio.features.rasterize(((g, 255) for g in water_features),
                                        out_shape=raster_dataset.shape,
                                        transform=raster_dataset.transform)

    save_image(cache_path, bitmap_image, raster_dataset)

    # TODO: Don't resize image but change create_tiles.
    bitmap_image = np.reshape(bitmap_image, (bitmap_image.shape[0], bitmap_image.shape[1], 1))
    bitmap_image[bitmap_image == 255] = 1
    return bitmap_image

def visualise_features(features, tile_size, out_path):
    get_path = lambda x: x[2]
    sorted_by_path = sorted(features, key=get_path)
    for path, predictions in itertools.groupby(sorted_by_path, get_path):
        satellite_img_name = get_file_name(path)
        satellite_file_name = "{}_wgs84.tif".format(satellite_img_name)
        path_wgs84 = os.path.join(WGS84_DIR, satellite_file_name)
        raster_dataset = rasterio.open(path_wgs84)
        bitmap_shape = (raster_dataset.shape[0], raster_dataset.shape[1], 1)
        bitmap = image_from_tiles(predictions, tile_size, bitmap_shape)
        bitmap = np.reshape(bitmap, (bitmap.shape[0], bitmap.shape[1]))
        out_file_name = "{}.tif".format(satellite_img_name)
        out = os.path.join(out_path, out_file_name)
        overlay_bitmap(bitmap, raster_dataset, out)


def save_tiles(file_path, tiled_features, tiled_labels):
    print("Store tile data at {}.".format(file_path))
    with open(file_path, "wb") as out:
        pickle.dump({"features": tiled_features, "labels": tiled_labels}, out)


def save_image(file_path, image, source):
    print("Save result at {}.".format(file_path))
    with rasterio.open(
            file_path, 'w',
            driver='GTiff',
            dtype=rasterio.uint8,
            count=1,
            width=source.width,
            height=source.height,
            transform=source.transform) as dst:
        dst.write(image, indexes=1)


def get_file_name(file_path):
    basename = os.path.basename(file_path)
    # Make sure we don't include the file extension.
    return os.path.splitext(basename)[0]
