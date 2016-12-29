"""Module to handle processing the raw GeoTIFF data from satellite imagery."""

import rasterio
import numpy as np


def read_geotiff(file_name, satellite_type="sentinel-2"):
    """TODO: Docstring."""
    raster_dataset = rasterio.open(file_name)
    bands = [raster_dataset.read(band_number)
             for band_number in raster_dataset.indexes]
    bands = np.dstack(bands)
    return raster_dataset, bands


def get_bounds(raster_dataset):
    pass


def create_tiles(raster_dataset, tile_size):
    pass


def create_bitmap(raster_dataset):
    """Pseudo algorithm:
    bounds = get_bounds(raster_dataset)
    water_features = extract_water(bounds)
    bitmap = zeroes(raster_dataset.shape)
    for feature in water_features:
        bitmap = add_feature(bitmap, feature)
    return bitmap
    """
    pass

def add_feature(bitmap, feature):
    """Pseudo algorithm:
    polygon = Polygon(feature)
    points_in_polygon = get_points_in_polygon(polygon)
    for point in polygon:
        bitmap[point] = 1
    return bitmap
    """
    pass
