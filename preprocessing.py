import fiona
import pickle
import rasterio
import rasterio.features
import time
import os
from config import TILES_DIR
from coordinate_translation import lat_lon_to_raster_crs
from process_geotiff import read_geotiff, read_bitmap, create_tiles, image_from_tiles, overlay_bitmap
import numpy as np


def preprocess_data(tile_size, dataset):

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
    satellite_img_name = get_file_name(geotiff_path)
    cache_path = "{}{}_{}.pickle".format(
        TILES_DIR, satellite_img_name, tile_size)
    try:
        print("Load from cache.")
        with open(cache_path) as f:
            tiles = pickle.open(f)

        return tiles["features"], tiles["labels"]
    except IOError as e:
        print("Cache not available. Compute tiles.")

    dataset, bands = read_geotiff(geotiff_path)

    water_img = create_image(dataset, shapefile_paths, geotiff_path)
    water_img[water_img == 255] = 1

    tiled_bands = create_tiles(bands, tile_size)
    tiled_bitmap = create_tiles(water_img, tile_size)

    save_tiles(cache_path, tiled_bands, tiled_bitmap)

    return tiled_bands, tiled_bitmap


def create_image(raster_dataset, shapefile_paths, satellite_path):
    # TODO: Better naming.
    satellite_img_name = get_file_name(satellite_path)
    water_img_cache = "{}{}_water.tif".format(
        WATER_BITMAPS_DIR, satellite_img_name)
    try:
        print("Load water image from cache.")
        _, water_img = read_geotiff(water_img_cache)
    except IOError as e:
        print("No cache file found.")

    features = np.empty((0,))

    print("Create image for water features.")
    for shapefile_path in shapefile_paths:
        print("Load shapefile {}.".format(shapefile_path))
        with fiona.open(shapefile_path) as shapefile:
            geometries = [feature['geometry'] for feature in shapefile]

            water_features = transform_coordinates(
                geometries, raster_dataset, shapefile_path)

            features = np.concatenate((features, water_features), axis=0)

    image = rasterio.features.rasterize(((g, 255) for g in features),
                                        out_shape=raster_dataset.shape,
                                        transform=raster_dataset.transform)

    save_image(water_img_cache, image, raster_dataset)
    image = np.reshape(image, (image.shape[0], image.shape[1], 1))
    return image


def transform_coordinates(geometries, dataset, shapefile_path):
    shapefile_name = get_file_name(shapefile_path)
    cache_path = "{}{}_{}_water_polygons.npy".format(WATER_POLYGONS_DIR,
                                                     shapefile_name,
                                                     dataset.crs['init'])
    try:
        print("Load coordinates from cache.")
        geometries = np.load(cache_path)
        return geometries
    except IOError as e:
        print("No cache file found.")

    print("Start computing coordinates.")
    t0 = time.time()
    for i, feature in enumerate(geometries):
        if feature['type'] == 'Polygon':
            for j, points in enumerate(feature['coordinates']):
                transformed_points = map(
                    lambda (lon, lat): lat_lon_to_raster_crs((lat, lon), dataset), points)
                geometries[i]['coordinates'][j] = transformed_points
        else:
            for j, ps in enumerate(feature['coordinates']):
                for k, points in enumerate(ps):
                    transformed_points = map(
                        lambda (lon, lat): lat_lon_to_raster_crs((lat, lon), dataset), points)
                    geometries[i]['coordinates'][j][k] = transformed_points
    t1 = time.time()
    print("Finished computing coordinates. Took {}s".format(t1 - t0))

    np.save(cache_path, geometries)
    return geometries


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

"""
if __name__ == '__main__':
    tile_size = 10
    geotiffs = [(muenster_img, [muenster_water])]
    features = []
    labels = []
    water_img = None
    for geotiff, shapefiles in geotiffs:
        dataset, bands = read_geotiff(geotiff)
        water_img = create_image(dataset, shapefiles, geotiff)
        water_img[water_img == 255] = 1
        tiled_bands = create_tiles(bands, tile_size)
        tiled_bitmap = create_tiles(water_img, tile_size)
        features += tiled_bands
        labels += tiled_bitmap
"""
