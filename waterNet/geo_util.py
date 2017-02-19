"""Functions used to process the GeoTIFF satellite images."""

import rasterio
import rasterio.warp
import fiona
import os
import itertools
import numpy as np
from io_util import get_file_name
from config import WGS84_DIR


def reproject_dataset(geotiff_path):
    """Project a GeoTIFF to the WGS84 coordinate reference system.
    See https://mapbox.github.io/rasterio/topics/reproject.html"""

    # We want to project the GeoTIFF coordinate reference system (crs)
    # to WGS84 (e.g. into the familiar Lat/Lon pairs). WGS84 is analogous
    # to EPSG:4326
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

        return rasterio.open(out_path), out_path


def create_tiles(bands_data, tile_size, path_to_geotiff):
    """Tile the satellite image which is given as a matrix into tiles of
    the given size.
    From https://github.com/trailbehind/DeepOSM."""

    rows, cols = bands_data.shape[0], bands_data.shape[1]

    all_tiled_data = []

    # Cartesian prodect of all the possible row and colum indexes. This
    # gives all possible left-upper positions of our tiles.
    tile_indexes = itertools.product(
        range(0, rows, tile_size), range(0, cols, tile_size))

    for (row, col) in tile_indexes:
        in_bounds = row + tile_size < rows and col + tile_size < cols
        if in_bounds:
            new_tile = bands_data[row:row + tile_size, col:col + tile_size]
            # Additionaly to the tile we also store its position, given by
            # its upper left corner, and the path to the GeoTIFF it belongs
            # to. We need this information to visualise our results later on.
            all_tiled_data.append((new_tile, (row, col), path_to_geotiff))

    return all_tiled_data


def image_from_tiles(tiles, tile_size, image_shape):
    """'Stitch' several tiles back together to form one image."""

    image = np.zeros(image_shape, dtype=np.uint8)

    for tile, (row, col), _ in tiles:
        tile = np.reshape(tile, (tile_size, tile_size))
        image[row:row + tile_size, col:col + tile_size] = tile

    return image


def overlay_bitmap(bitmap, raster_dataset, out_path, color='blue'):
    """Overlay the given satellite image with a bitmap."""

    # RGB values for possible color options.
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
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(red, 1)
        dst.write(green, 2)
        dst.write(blue, 3)

    return rasterio.open(out_path)


def create_shapefile(bitmap, raster_dataset, out_path):
    """TODO: Docstring."""

    shapes = rasterio.features.shapes(bitmap, transform=raster_dataset.transform)
    records = map(lambda (geom, _): {"geometry": geom, "properties": {}}, shapes)
    schema = {
        "geometry": "Polygon",
        "properties": {}
    }
    with fiona.open(out_path, 'w', driver="ESRI Shapefile", crs=raster_dataset.crs, schema=schema) as f:
        f.writerecords(records)


def visualise_labels(labels, tile_size, out_path):
    """Given the labels of a satellite image as tiles. Overlay the source image with the labels
    to check if labels are roughly correct."""

    # The tiles might come from different satellite images so we have to
    # group them according to their source image.
    get_path = lambda (tiles, pos, path): path
    sorted_by_path = sorted(labels, key=get_path)
    for path, predictions in itertools.groupby(sorted_by_path, get_path):
        raster_dataset = rasterio.open(path)

        bitmap_shape = (raster_dataset.shape[0], raster_dataset.shape[1])
        bitmap = image_from_tiles(predictions, tile_size, bitmap_shape)

        satellite_img_name = get_file_name(path)
        out_file_name = "{}.tif".format(satellite_img_name)
        out = os.path.join(out_path, out_file_name)
        overlay_bitmap(bitmap, raster_dataset, out)

def visualise_results(results, tile_size, out_path, out_format="GeoTIFF"):
    """Given the predictions, false positves and the labels of our model visualise them on the satellite
    image they belong to."""

    # The tiles of the predictions, false positives and labels are all in "results".
    # We need ways to get extract them individually to pass them to overlay_bitmap.
    get_predictions = lambda (tiles, pos, path): (tiles[0], pos, path)
    get_labels = lambda (tiles, pos, path): (tiles[1], pos, path)
    get_false_positives =  lambda (tiles, pos, path): (tiles[2], pos, path)

    get_path = lambda (tiles,pos , path): path
    sorted_by_path = sorted(results, key=get_path)
    for path, result_tiles in itertools.groupby(sorted_by_path, get_path):
        raster_dataset = rasterio.open(path)
        dataset_shape = (raster_dataset.shape[0], raster_dataset.shape[1])

        result_tiles = list(result_tiles)
        predictions = map(get_predictions, result_tiles)
        labels = map(get_labels, result_tiles)
        false_positives = map(get_false_positives, result_tiles)

        satellite_img_name = get_file_name(path)
        file_extension = "tif" if out_format == "GeoTIFF" else "shp"
        out_file_name = "{}_results.{}".format(satellite_img_name, file_extension)
        out = os.path.join(out_path, out_file_name)

        if out_format == "GeoTIFF":
            # We first write the labels in blue, then predictions in green and then false positives in red.
            # This way the true positives will be green, false positives red, false negatives blue and everything
            # else in the image will be true negatives.
            for tiles, color in [(labels, 'blue'), (predictions, 'green'), (false_positives, 'red')]:
                bitmap = image_from_tiles(tiles, tile_size, dataset_shape)
                raster_dataset = overlay_bitmap(bitmap, raster_dataset, out, color=color)
        elif out_format == "Shapefile":
            bitmap = image_from_tiles(predictions, tile_size, dataset_shape)
            create_shapefile(bitmap, raster_dataset, out)
