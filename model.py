"""Model used to predict water."""

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
import numpy as np
from process_geotiff import read_geotiff, read_bitmap, create_tiles
from transform_shapefile import create_image, muenster_img, muenster_water, DATA_DIR

MODEL_CACHE = DATA_DIR + "working/logs/"

def normalize_input(features):
    features = features.astype(np.float32)
    return np.multiply(features, 1.0 / 255.0)

def train_model(features, labels, tile_size):
    features = normalize_input(features)
    model = create_model(features, tile_size)
    print("Start training.")
    model.fit(features, labels) # TODO: Hyperparameters.

def create_model(features, tile_size):
    # TODO: Declare 3 a variable as num_channels.
    # From DeepOSM.
    net = tflearn.input_data(shape=[None, tile_size, tile_size, 3])
    net = conv_2d(net, 64, 12, strides=4, activation='relu')
    net = max_pool_2d(net, 3)

    softmax = tflearn.fully_connected(net, tile_size*tile_size, activation='softmax')

    momentum = tflearn.optimizers.Momentum(
        learning_rate=0.005, momentum=0.9,
        lr_decay=0.002, name='Momentum'
    )

    net = tflearn.regression(softmax, optimizer=momentum, loss='categorical_crossentropy')

    return tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir=MODEL_CACHE)


def evaluate_model(model, features, labels):
    pass

if __name__ == '__main__':
    tile_size = 64
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

    labels = np.reshape(labels, (labels.shape[0], tile_size*tile_size))
    train_model(features, labels, tile_size)
