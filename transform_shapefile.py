import fiona
import rasterio
import rasterio.features
import time
import os
from coordinate_translation import lat_lon_to_raster_crs
from process_geotiff import read_geotiff, read_bitmap, create_tiles
import numpy as np

DATA_DIR = "/Users/Tim/dev/python/DeepWater/data/"
CACHE_DIR = DATA_DIR + "working/cache/"
SENTINEL_DIR = DATA_DIR + "input/Sentinel-2/"
SHAPEFILE_DIR = DATA_DIR + "input/Shapefiles/"

netherlands_water = DATA_DIR + "input/Shapefiles/netherlands-latest-free/gis.osm_water_a_free_1.shp"
muenster_water = DATA_DIR + "input/Shapefiles/muenster-regbez-latest-free/gis.osm_water_a_free_1.shp"

netherlands_shapefile = "netherlands-latest-free/gis.osm_water_a_free_1"
muenster_shapefile = "muenster-regbez-latest-free/gis.osm_water_a_free_1"

netherlands_img_name = "S2A_OPER_MSI_L1C_TL_SGS__20160908T110617_20160908T161324_A006340_T31UFU_N02_04_01"
muenster_img_name = "S2A_OPER_MSI_L1C_TL_SGS__20161204T105758_20161204T143433_A007584_T32ULC_N02_04_01"

netherlands_img = SENTINEL_DIR + netherlands_img_name + ".tif"
muenster_img = SENTINEL_DIR + muenster_img_name + ".tif"

netherlands_cache = CACHE_DIR + netherlands_img_name + "_water_polygons.npy"
muenster_cache = CACHE_DIR + muenster_img_name + "_water_polygons.npy"

nl_result_img = CACHE_DIR + netherlands_img_name + "_water.tif"
ms_result_img = CACHE_DIR + muenster_img_name + "_water.tif"


def transfrom_coordinates(geometries, dataset, cache_path):
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


def create_image(raster_dataset, shapefile_paths, satellite_path):
    # TODO: Better naming.
    # TODO: Load from cache.
    satellite_img_name = get_file_name(satellite_path)
    water_img_cache = "{}{}_water.tif".format(CACHE_DIR, satellite_img_name)

    try:
        print("Load water image from cache.")
        _, water_img = read_geotiff(water_img_cache)
        return water_img
    except IOError as e:
        print("No cache file found.")

    features = np.empty((0,))

    print("Create image for water features.")
    for shapefile_path in shapefile_paths:
        print("Load shapefile {}.".format(shapefile_path))
        with fiona.open(shapefile_path) as shapefile:
            geometries = [feature['geometry'] for feature in shapefile]
            shapefile_name = get_file_name(shapefile_path)
            cache_path = "{}{}_{}_water_polygons.npy".format(CACHE_DIR,
                                                             shapefile_name,
                                                             raster_dataset.crs['init'])
            water_features = transfrom_coordinates(geometries, raster_dataset, cache_path)
            features = np.concatenate((features, water_features), axis=0)

    image = rasterio.features.rasterize(((g, 255) for g in features),
                                        out_shape=raster_dataset.shape,
                                        transform=raster_dataset.transform)

    save_image(water_img_cache, image, raster_dataset)

    image = np.reshape(image, (image.shape[0], image.shape[1], 1))
    return image


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

if __name__ == '__main__':
    tile_size = 10
    tile_overlap = 0
    geotiffs = [(muenster_img, [muenster_water])]
    features = np.empty((0,tile_size,tile_size,3))
    labels = np.empty((0,tile_size,tile_size,1)) # TODO: Check shape
    for geotiff, shapefiles in geotiffs:
        dataset, bands = read_geotiff(geotiff)
        water_img = create_image(dataset, shapefiles, geotiff)
        water_img[water_img == 255] = 1
        tiled_bands = create_tiles(bands, tile_size, tile_overlap)
        tiled_bitmap = create_tiles(water_img, tile_size, tile_overlap)
        features = np.concatenate((features, tiled_bands), axis=0)
        labels = np.concatenate((labels, tiled_bitmap), axis=0)
