from os.path import join

DATA_DIR = "/Users/Tim/dev/python/DeepWater/data/"
TRAIN_DATA_DIR = join(DATA_DIR, "working", "train_data")
SENTINEL_DIR = join(DATA_DIR, "input", "Sentinel-2")
SHAPEFILE_DIR = join(DATA_DIR, "input", "Shapefiles")

TILES_DIR = join(TRAIN_DATA_DIR, "tiles")
WATER_BITMAPS_DIR = join(TRAIN_DATA_DIR, "water_bitmaps")
WGS84_DIR = join(TRAIN_DATA_DIR, "WGS84_images")

WATER_POLYGONS_DIR = join(DATA_DIR, "working", "water_polygons")
MODELS_DIR = join(DATA_DIR, "working", "models")

SHAPEFILE_CHECKPOINTS_DIR = join(WATER_POLYGONS_DIR, "checkpoints")

OUTPUT_DIR = join(DATA_DIR, "output")
LOGS_DIR = join(OUTPUT_DIR, "logs")
TENSORBOARD_DIR = join(OUTPUT_DIR, "tensorboard")

MUENSTER_SHAPEFILE = join(DATA_DIR, "input", "Shapefiles", "muenster-regbez-latest-free", "gis.osm_water_a_free_1.shp")
NETHERLANDS_SHAPEFILE = join(DATA_DIR, "input", "Shapefiles", "netherlands-latest-free", "gis.osm_water_nl_free_1.shp")
NRW_SHAPEFILE = join(DATA_DIR, "input", "Shapefiles", "nordrhein-westfalen-latest-free", "gis.osm_water_nrw_free_1.shp")
OCEAN_SHAPEFILE = join(DATA_DIR, "input", "Shapefiles", "water-polygons-split-4326", "ocean_polygons.shp")
SWITZERLAND_SHAPEFILE = join(SHAPEFILE_DIR, "switzerland-latest-free", "gis.osm_water_a_free_1.shp")
FRANCHE_COMTE_SHAPEFILE = join(SHAPEFILE_DIR, "franche-comte-latest-free", "gis.osm_water_a_free_1.shp")
RHONE_ALPES_SHAPEFILE = join(SHAPEFILE_DIR, "rhone-alpes-latest-free", "gis.osm_water_a_free_1.shp")

MUENSTER_SATELLITE = join(SENTINEL_DIR, "S2A_OPER_MSI_L1C_TL_SGS__20161204T105758_20161204T143433_A007584_T32ULC_N02_04_01.tif")
AMSTERDAM_SATELLITE = join(SENTINEL_DIR, "S2A_OPER_MSI_L1C_TL_SGS__20160908T110617_20160908T161324_A006340_T31UFU_N02_04_01.tif")
LAUSANNE_SATELLITE = join(SENTINEL_DIR, "S2A_OPER_MSI_L1C_TL_SGS__20161022T104617_20161022T174222_A006969_T31TGM_N02_04_01.tif")

SENTINEL_DATASET_TRAIN = [(AMSTERDAM_SATELLITE, [NETHERLANDS_SHAPEFILE, OCEAN_SHAPEFILE]),
                          (MUENSTER_SATELLITE, [NRW_SHAPEFILE, NETHERLANDS_SHAPEFILE])]
SENTINEL_DATASET_TEST = [(LAUSANNE_SATELLITE, [SWITZERLAND_SHAPEFILE, FRANCHE_COMTE_SHAPEFILE, RHONE_ALPES_SHAPEFILE, OCEAN_SHAPEFILE])]
SENTINEL_DATASET = {
    "train": SENTINEL_DATASET_TRAIN,
    "test": SENTINEL_DATASET_TEST
}

DEBUG_DATASET = {
    "train": [(MUENSTER_SATELLITE, [MUENSTER_SHAPEFILE])],
    "test": []
}
