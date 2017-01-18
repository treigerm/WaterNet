"""Implementation of the convolutional neural net."""

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard
import os
import numpy as np
from config import MODELS_DIR, TENSORBOARD_DIR
from io_util import save_makedirs, save_model


def train_model(model,
                features,
                labels,
                tile_size,
                model_id,
                nb_epoch=10,
                checkpoints=False,
                tensorboard=False):
    """Train a model with the given features and labels."""

    # The features and labels are a list of triples when passed
    # to the function. Each triple contains the tile and information
    # about its source image and its postion in the source. To train
    # the model we extract just the tiles.
    X, y = get_matrix_form(features, labels, tile_size)

    X = normalise_input(X)

    # Directory which is used to store the model and its weights.
    model_dir = os.path.join(MODELS_DIR, model_id)

    checkpointer = None
    if checkpoints:
        checkpoints_file = os.path.join(model_dir, "weights.hdf5")
        checkpointer = ModelCheckpoint(checkpoints_file)

    tensorboarder = None
    if tensorboard:
        log_dir = os.path.join(TENSORBOARD_DIR, model_id)
        tensorboarder = TensorBoard(log_dir=log_dir)

    callbacks = [c for c in [checkpointer, tensorboarder] if c]

    print("Start training.")
    model.fit(X, y, nb_epoch=nb_epoch, callbacks=callbacks, validation_split=0.1)

    save_model(model, model_dir)
    return model


def init_model(tile_size,
               model_id,
               architecture='one_layer',
               nb_filters_1=64,
               filter_size_1=12,
               stride_1=(4, 4),
               pool_size_1=(3, 3),
               nb_filters_2=128,
               filter_size_2=4,
               stride_2=(1, 1),
               learning_rate=0.005,
               momentum=0.9,
               decay=0.002):
    """Initialise a new model with the given hyperparameters and save it for later use."""

    num_channels = 3

    model = Sequential()

    if architecture == 'one_layer':
        model.add(
            Convolution2D(
                nb_filters_1,
                filter_size_1,
                filter_size_1,
                subsample=stride_1,
                input_shape=(tile_size, tile_size, num_channels)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size_1))
        model.add(Flatten())
        model.add(Dense(tile_size * tile_size))
        model.add(Activation('sigmoid'))
    elif architecture == 'two_layer':
        model.add(
            Convolution2D(
                nb_filters_1,
                filter_size_1,
                filter_size_1,
                subsample=stride_1,
                input_shape=(tile_size, tile_size, num_channels)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size_1))
        model.add(
            Convolution2D(
                nb_filters_2,
                filter_size_2,
                filter_size_2,
                subsample=stride_2))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(tile_size * tile_size))
        model.add(Activation('sigmoid'))


    model = compile_model(model, learning_rate, momentum, decay)

    # Print a summary of the model to the console.
    model.summary()

    model_dir = os.path.join(MODELS_DIR, model_id)
    save_makedirs(model_dir)

    save_model(model, model_dir)

    return model


def compile_model(model, learning_rate, momentum, decay):
    """Compile the keras model with the given hyperparameters."""

    optimizer = SGD(lr=learning_rate, momentum=momentum, decay=decay)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])

    return model

def normalise_input(features):
    """Normalise the features such that all values are in the range [0,1]."""
    features = features.astype(np.float32)

    return np.multiply(features, 1.0 / 255.0)


def get_matrix_form(features, labels, tile_size):
    """Transform a list of triples of features and labels. To a matrix which contains
    only the tiles used for training the model."""
    features = [tile for tile, position, path in features]
    labels = [tile for tile, position, path in labels]

    # The model will have one output corresponding to each pixel in the feature tile.
    # So we need to transform the labels which are given as a 2D bitmap into a vector.
    labels = np.reshape(labels, (len(labels), tile_size * tile_size))
    return np.array(features), np.array(labels)
