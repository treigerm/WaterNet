"""Module to handle processing the raw GeoTIFF data from satellite imagery."""

import rasterio
import rasterio.warp
import os
import itertools
import numpy as np
from io_util import get_file_name
from config import WGS84_DIR


def read_geotiff(file_name):
    """TODO: Docstring."""
    raster_dataset = rasterio.open(file_name)
    bands = read_bands(raster_dataset)
    return raster_dataset, bands


def read_bands(raster_dataset):
    bands = [
        raster_dataset.read(band_number)
        for band_number in raster_dataset.indexes
    ]
    bands = np.dstack(bands)
    return bands


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


def create_tiles(bands_data, tile_size, path_to_geotiff):
    """From https://github.com/trailbehind/DeepOSM."""
    # TODO: Select bands.

    rows, cols, n_bands = bands_data.shape

    all_tiled_data = []

    tile_indexes = itertools.product(
        range(0, rows, tile_size), range(0, cols, tile_size))

    for (row, col) in tile_indexes:
        in_bounds = row + tile_size < rows and col + tile_size < cols
        if in_bounds:
            new_tile = bands_data[row:row + tile_size, col:col + tile_size,
                                  0:n_bands]
            all_tiled_data.append((new_tile, (row, col), path_to_geotiff))

    return all_tiled_data


def image_from_tiles(tiles, tile_size, image_shape):
    image = np.zeros(image_shape, dtype=np.uint8)

    for tile, (row, col), _ in tiles:
        image[row:row + tile_size, col:col + tile_size, :] = tile

    return image


def overlay_bitmap(bitmap, raster_dataset, out_path, color='blue'):
    colors = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255)
    }
    red, green, blue = raster_dataset.read()
    red[bitmap == 1] = colors[color][0]
    green[bitmap == 1] = colors[color][1]
    blue[bitmap == 1] = colors[color][2]
    profile = raster_dataset.profile
    dst = rasterio.open(out_path, 'w', **profile)
    #with rasterio.open(out_path, 'w', **profile) as dst:
    dst.write(red, 1)
    dst.write(green, 2)
    dst.write(blue, 3)
    dst.close()
    return rasterio.open(out_path)


def visualise_features(features, tile_size, out_path):
    get_path = lambda x: x[2]
    sorted_by_path = sorted(features, key=get_path)
    for path, predictions in itertools.groupby(sorted_by_path, get_path):
        satellite_img_name = get_file_name(path)
        satellite_file_name = "{}_wgs84.tif".format(satellite_img_name)
        path_wgs84 = os.path.join(WGS84_DIR, satellite_file_name)
        raster_dataset = rasterio.open(path_wgs84)
        # TODO: Don' reshape bitmap.
        bitmap_shape = (raster_dataset.shape[0], raster_dataset.shape[1], 1)
        bitmap = image_from_tiles(predictions, tile_size, bitmap_shape)
        bitmap = np.reshape(bitmap, (bitmap.shape[0], bitmap.shape[1]))
        out_file_name = "{}.tif".format(satellite_img_name)
        out = os.path.join(out_path, out_file_name)
        overlay_bitmap(bitmap, raster_dataset, out)

def visualise_results(results, tile_size, out_path):
    get_path = lambda x: x[2]
    get_predictions = lambda x: (x[0][0], x[1], x[2])
    get_labels = lambda x: (x[0][1], x[1], x[2])
    get_false_positives = lambda x: (x[0][2], x[1], x[2])
    sorted_by_path = sorted(results, key=get_path)
    for path, result_tiles in itertools.groupby(sorted_by_path, get_path):
        satellite_img_name = get_file_name(path)
        satellite_file_name = "{}_wgs84.tif".format(satellite_img_name)
        path_wgs84 = os.path.join(WGS84_DIR, satellite_file_name)
        raster_dataset = rasterio.open(path_wgs84)
        bitmap_shape = (raster_dataset.shape[0], raster_dataset.shape[1], 1)
        result_tiles = list(result_tiles)
        predictions = map(get_predictions, result_tiles)
        labels = map(get_labels, result_tiles)
        false_positives = map(get_false_positives, result_tiles)
        out_file_name = "{}.tif".format(satellite_img_name)
        out = os.path.join(out_path, out_file_name)
        for tiles, color in [(labels, 'blue'), (predictions, 'green'), (false_positives, 'red')]:
            bitmap = image_from_tiles(tiles, tile_size, bitmap_shape)
            bitmap = np.reshape(bitmap, (bitmap.shape[0], bitmap.shape[1]))
            raster_dataset = overlay_bitmap(bitmap, raster_dataset, out, color=color)


