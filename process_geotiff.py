"""Module to handle processing the raw GeoTIFF data from satellite imagery."""

import rasterio
import rasterio.warp
import Shapely
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


def create_tiles(raster_dataset, tile_size):
    pass


def create_bitmap(raster_dataset):
    """TODO: Docstring"""
    bounds = get_bounds(raster_dataset)
    water_features = extract_water(bounds)
    bitmap = np.zeros(raster_dataset.shape, dtype=np.int)
    for feature in water_features:
        feature = [ lat_lon_to_pixel(point, raster_dataset) for point in feature ]
        bitmap = add_feature(bitmap, feature)
    return bitmap


# TODO: Different feature adding for way and relation.
def add_feature(bitmap, feature):
    """TODO: Docstring"""
    polygon = Shapely.Polygon(feature)
    points_in_polygon = get_points_in_polygon(polygon)
    for x, y in points_in_polygon:
        # TODO: Check if x y in bounds
        bitmap[y][x] = 1
    return bitmap

def get_points_in_polygon(polygon):
    minx, miny, maxx, maxy = polygon.bounds
    possible_points = itertools.product(range(minx, maxx+1), range(miny, maxy+1))
    return [ point for point in possible_points if polygon.contains(point) ]
