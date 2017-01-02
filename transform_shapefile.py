import fiona
import rasterio
import rasterio.features
import time
from coordinate_translation import lat_lon_to_pixel
import numpy as np

DATA_DIR = "/Users/Tim/dev/python/DeepWater/data"
netherlands_cache = "/working/cache/water_netherlands.npy"
muenster_cache = "/working/cache/water_muenster.npy"

def transfrom_coordinates(geometries, dataset, cache_name):
    try:
        print("Try to load from cache.")
        geometries = np.load(DATA_DIR + cache_name)
        return geometries
    except IOError as e:
        print("No cache file found.")

    print("Start computing pixel locations.")
    t0 = time.time()
    for i, feature in enumerate(geometries):
        if feature['type'] == 'Polygon':
            for j, points in enumerate(feature['coordinates']):
                transformed_points = map(lambda (lon, lat): lat_lon_to_pixel((lat, lon), dataset), points)
                geometries[i]['coordinates'][j] = transformed_points
        else:
            for j, ps in enumerate(feature['coordinates']):
                for k, points in enumerate(ps):
                    transformed_points = map(lambda (lon, lat): lat_lon_to_pixel((lat, lon), dataset), points)
                    geometries[i]['coordinates'][j][k] = transformed_points
    t1 = time.time()
    print("Finished computing pixel locations. Took {}s".format(t1 - t0))

    np.save(DATA_DIR + cache_name, geometries)
    return geometries

water_netherlands = "/input/Shapefiles/netherlands-latest-free/gis.osm_water_a_free_1.shp"
water_muenster = "/input/Shapefiles/muenster-regbez-latest-free/gis.osm_water_a_free_1.shp"
with fiona.open(DATA_DIR + water_muenster) as shapefile:
    features = [feature['geometry'] for feature in shapefile]

netherlands_img = "/input/Sentinel-2/S2A_OPER_MSI_L1C_TL_SGS__20160908T110617_20160908T161324_A006340_T31UFU_N02_04_01.tif"
muenster_img = "/input/Sentinel-2/S2A_OPER_MSI_L1C_TL_SGS__20161204T105758_20161204T143433_A007584_T32ULC_N02_04_01.tif"
with rasterio.open(DATA_DIR + muenster_img) as dataset:
    features = transfrom_coordinates(features, dataset, muenster_cache)
    image = rasterio.features.rasterize(((g, 255) for g in features),
                                        out_shape=dataset.shape)
    transform = dataset.transform
    width, height = dataset.width, dataset.height

nl_result_img = "/working/netherlands_water.tif"
ms_result_img = "/working/muenster_water.tif"
with rasterio.open(
        DATA_DIR + ms_result_img, 'w',
        driver='GTiff',
        dtype=rasterio.uint8,
        count=1,
        width=width,
        height=height) as dst:
    dst.write(image, indexes=1)
