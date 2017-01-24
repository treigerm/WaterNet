import os
from os.path import join

# Specify where the program can find the data.
DATA_DIR = os.environ["DATA_DIR"]


# Directories which contain satellite imagery and shapefiles.
SENTINEL_DIR = join(DATA_DIR, "input", "Sentinel-2")
SHAPEFILE_DIR = join(DATA_DIR, "input", "Shapefiles")

# Directories to store everything related to the training data.
TRAIN_DATA_DIR = join(DATA_DIR, "working", "train_data")
TILES_DIR = join(TRAIN_DATA_DIR, "tiles")
WATER_BITMAPS_DIR = join(TRAIN_DATA_DIR, "water_bitmaps")
WGS84_DIR = join(TRAIN_DATA_DIR, "WGS84_images")
LABELS_DIR = join(TRAIN_DATA_DIR, "labels_images")

# Directory to store the models and weights.
MODELS_DIR = join(DATA_DIR, "working", "models")

# Directories for model outputs (metrics, visualisations, etc).
OUTPUT_DIR = join(DATA_DIR, "output")
TENSORBOARD_DIR = join(OUTPUT_DIR, "tensorboard")

# Paths to the shapefiles.
MUENSTER_SHAPEFILE = join(SHAPEFILE_DIR, "muenster-regbez-latest-free", "gis.osm_water_a_free_1.shp")
NETHERLANDS_SHAPEFILE = join(SHAPEFILE_DIR, "netherlands-latest-free", "gis.osm_water_a_free_1.shp")
NRW_SHAPEFILE = join(SHAPEFILE_DIR, "nordrhein-westfalen-latest-free", "gis.osm_water_a_free_1.shp")
OCEAN_SHAPEFILE = join(SHAPEFILE_DIR, "water-polygons-split-4326", "water_polygons.shp")
BAYERN_SHAPEFILE = join(SHAPEFILE_DIR, "bayern-latest-free", "gis.osm_water_a_free_1.shp")
ENGLAND_SHAPEFILE = join(SHAPEFILE_DIR, "england-latest-free", "gis.osm_water_a_free_1.shp")
HUNGARY_SHAPEFILE = join(SHAPEFILE_DIR, "hungary-latest-free", "gis.osm_water_a_free_1.shp")
VENICE_SHAPEFILE = join(SHAPEFILE_DIR, "nord-est-latest-free", "gis.osm_water_a_free_1.shp")

# Paths to the satellite images.
MUENSTER_SATELLITE = join(SENTINEL_DIR, "S2A_OPER_MSI_L1C_TL_SGS__20161204T105758_20161204T143433_A007584_T32ULC_N02_04_01.tif")
AMSTERDAM_SATELLITE = join(SENTINEL_DIR, "S2A_OPER_MSI_L1C_TL_SGS__20160908T110617_20160908T161324_A006340_T31UFU_N02_04_01.tif")
MUNICH_SATELLITE = join(SENTINEL_DIR, "S2A_OPER_MSI_L1C_TL_SGS__20160929T103545_20160929T154211_A006640_T32UPU_N02_04_01.tif")
LIVERPOOL_SATELLITE = join(SENTINEL_DIR, "S2A_OPER_MSI_L1C_TL_SGS__20160719T112949_20160719T165219_A005611_T30UVE_N02_04_01.tif")
LONDON_SATELLITE = join(SENTINEL_DIR, "L1C_T30UXC_A007999_20170102T111441.tif")
BUDAPEST_SATELLITE = join(SENTINEL_DIR, "S2A_OPER_MSI_L1C_TL_SGS__20161129T100308_20161129T134154_A007512_T33TYN_N02_04_01.tif")
VENICE_SATELLITE = join(SENTINEL_DIR, "S2A_OPER_MSI_L1C_TL_SGS__20161115T101822_20161115T171903_A007312_T32TQR_N02_04_01.tif")

# Define which shapefiles correspond to which satellite images.
SENTINEL_DATASET_TRAIN = [(AMSTERDAM_SATELLITE, [NETHERLANDS_SHAPEFILE, OCEAN_SHAPEFILE]),
                          (MUENSTER_SATELLITE, [NRW_SHAPEFILE, NETHERLANDS_SHAPEFILE]),
                          (MUNICH_SATELLITE, [BAYERN_SHAPEFILE]),
                          (LIVERPOOL_SATELLITE, [ENGLAND_SHAPEFILE, OCEAN_SHAPEFILE]),
                          (LONDON_SATELLITE, [ENGLAND_SHAPEFILE]),
                          (VENICE_SATELLITE, [VENICE_SHAPEFILE, OCEAN_SHAPEFILE])]
SENTINEL_DATASET_TEST = [(BUDAPEST_SATELLITE, [HUNGARY_SHAPEFILE])]
SENTINEL_DATASET = {
    "train": SENTINEL_DATASET_TRAIN,
    "test": SENTINEL_DATASET_TEST
}

# Small dataset used for debugging purposes.
DEBUG_DATASET = {
    "train": [(MUENSTER_SATELLITE, [MUENSTER_SHAPEFILE])],
    "test": []
}

DATASETS = {"sentinel": SENTINEL_DATASET, "debug": DEBUG_DATASET}

