"""Model used to predict water."""

import tflearn
import numpy as np

def normalize_input(features):
    features = features.astype(np.float32)
    return features.multiply(features, 1.0 / 255.0)

def train_model(features, labels, tile_size):
    pass

def create_model(features, labels, tile_size):
    # TODO: Declare 3 a variable as num_channels.
    # From DeepOSM.
    net = tflearn.input_data(shape=[None, tile_size, tile_size, 3])
    net = tflearn.conv2d(net, 64, 12, strides=4, activation='relu')
    net = max_pool_2d(net, 3)

    softmax = tflearn.fully_connected(net, 2, activation='softmax')

    momentum = tflearn.optimizers.Momentum(
        learning_rate=0.005, momentum=0.9,
        lr_decay=0.002, name='Momentum'
    )

    net = tflearn.regression(softmax, optimizer=momentum, loss='categorical_crossentropy')

    return tflearn.DNN(net, tensorboard_verbose=0)


def evaluate_model(model, features, labels):
    pass
