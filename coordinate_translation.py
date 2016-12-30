"""Translate latitude and longitude coordinates to pixels on a GeoTIFF and vice versa."""

import rasterio.warp
import rasterio.transform

def lat_lon_to_pixel(point, raster_dataset):
    lat, lon = point
    src_crs = "EPSG:4326"
    dst_crs = raster_dataset.crs
    xs, ys = rasterio.warp.transform(src_crs, dst_crs, [lat], [lon])
    rows, cols = rasterio.transform.rowcol(raster_dataset.transform, xs, ys)
    return (rows[0], cols[0])

def pixel_to_lat_lon(pixel, raster_dataset):
    x, y = raster_dataset.transform * pixel
    # Coordinate reference system of the GeoTIFF.
    src_crs = raster_dataset.crs
    # Destination coordinate reference system. We want to get the longitude
    # and latitude so we use EPSG:4326 also known as WGS 84.
    # TODO: Declare global
    dst_crs = "EPSG:4326"
    lats, lons = rasterio.warp.transform(src_crs, dst_crs, [x], [y])
    return (lats[0], lons[0])
