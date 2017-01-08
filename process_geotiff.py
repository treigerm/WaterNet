"""Module to handle processing the raw GeoTIFF data from satellite imagery."""

import rasterio
import rasterio.warp
import numpy as np
import itertools
from osm_feature_extraction import extract_water


def read_geotiff(file_name):
    """TODO: Docstring."""
    raster_dataset = rasterio.open(file_name)
    bands = [raster_dataset.read(band_number)
             for band_number in raster_dataset.indexes]
    bands = np.dstack(bands)
    return raster_dataset, bands


def get_bounds(raster_dataset):
    """TODO: Docstring."""
    # Coordinate reference system of the GeoTIFF.
    src_crs = raster_dataset.crs
    # Destination coordinate reference system. We want to get the longitude
    # and latitude so we use EPSG:4326 also known as WGS 84.
    dst_crs = "EPSG:4326"
    west, south, east, north = rasterio.warp.transform_bounds(
        src_crs, dst_crs, *raster_dataset.bounds)
    return {
        "west": west,
        "south": south,
        "east": east,
        "north": north
    }


def read_bitmap(file_name):
    """TODO: Docstring."""
    # TODO: Outputshape of bitmap?
    raster_dataset, bitmap = read_geotiff(file_name)
    bitmap[bitmap == 255] = 1
    return raster_dataset, bitmap


def create_tiles(bands_data, tile_size, path_to_geotiff):
    """From https://github.com/trailbehind/DeepOSM."""
    # TODO: Select bands.

    rows, cols, n_bands = bands_data.shape

    all_tiled_data = []

    for row in range(0, cols, tile_size):
        for col in range(0, rows, tile_size):
            in_bounds = row + tile_size < rows and col + tile_size < cols
            if in_bounds:
                new_tile = bands_data[row:row + tile_size, col:col + tile_size, 0:n_bands]
                all_tiled_data.append((new_tile, (row, col), path_to_geotiff)) 

    return all_tiled_data

def image_from_tiles(tiles, tile_size, image_shape):
    image = np.zeros(image_shape, dtype=np.uint8)

    for tile, (row, col), _ in tiles:
        image[row:row + tile_size, col:col + tile_size, :] = tile

    return image


def overlay_bitmap(bitmap, raster_dataset, out_path):
    # TODO: Choose color.
    red, green, blue = raster_dataset.read()
    red[bitmap == 1] = 0
    green[bitmap == 1] = 0
    blue[bitmap == 1] = 255
    profile = raster_dataset.profile
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(red, 1)
        dst.write(green, 2)
        dst.write(blue, 3)


def create_bitmap(raster_dataset):
    """TODO: Docstring"""
    bounds = get_bounds(raster_dataset)
    water_features = extract_water(bounds)
    bitmap = np.zeros(raster_dataset.shape, dtype=np.int)
    for feature in water_features:
        feature = [ lat_lon_to_pixel(point, raster_dataset) for point in feature ]
        bitmap = add_feature(bitmap, feature)
    return bitmap
