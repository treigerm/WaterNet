"""Module to handle processing the raw GeoTIFF data from satellite imagery."""

import rasterio
import rasterio.warp
import numpy as np
import itertools
from osm_feature_extraction import extract_water


def read_geotiff(file_name, satellite_type="sentinel-2"):
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


def create_tiles(bands_data, tile_size, tile_overlap):
    """From https://github.com/trailbehind/DeepOSM."""
    # TODO: Select bands.
    # TODO: Add cache and have tiled data store path to raster data.

    rows, cols, n_bands = bands_data.shape

    all_tiled_data = []

    # TODO: Rename to row and col
    for x in range(0, cols, tile_size - tile_overlap):
        for y in range(0, rows, tile_size - tile_overlap):
            in_bounds = x + tile_size < rows and y + tile_size < cols
            if in_bounds:
                new_tile = bands_data[x:x + tile_size, y:y + tile_size, 0:n_bands]
                all_tiled_data.append(new_tile)

    return all_tiled_data


def create_bitmap(raster_dataset):
    """TODO: Docstring"""
    bounds = get_bounds(raster_dataset)
    water_features = extract_water(bounds)
    bitmap = np.zeros(raster_dataset.shape, dtype=np.int)
    for feature in water_features:
        feature = [ lat_lon_to_pixel(point, raster_dataset) for point in feature ]
        bitmap = add_feature(bitmap, feature)
    return bitmap
