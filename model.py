"""Model used to predict water."""

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard
import pickle
import os
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from process_geotiff import visualise_features
from config import MODELS_DIR
from config import TENSORBOARD_DIR
from io_util import save_makedirs, save_model
plt.style.use('ggplot')


def normalize_input(features):
    features = features.astype(np.float32)

    return np.multiply(features, 1.0 / 255.0)


def train_model(model,
                features,
                labels,
                tile_size,
                model_id,
                nb_epoch=10,
                checkpoints=False,
                tensorboard=False):
    X, y = get_matrix_form(features, labels, tile_size)
    X = normalize_input(X)

    model_dir = os.path.join(MODELS_DIR, model_id)
    save_makedirs(model_dir)

    checkpointer = None
    if checkpoints:
        checkpoints_path = os.path.join(model_dir, "checkpoints")
        save_makedirs(checkpoints_path)
        checkpoints_file = os.path.join(checkpoints_path, "weights.hdf5")
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
               architecture='one_layer',
               nb_filters_1=64,
               filter_size_1=12,
               stride_1=(4, 4),
               pool_size_1=(3, 3),
               nb_filters_2=128,
               filter_size_2=4,
               stride_2=(1, 1)):

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


    momentum = SGD(lr=0.005, momentum=0.9, decay=0.002)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=momentum,
        metrics=['accuracy'])

    print(model.summary())

    return model


def evaluate_model(model, features, labels, tile_size, out_path):
    print('_' * 100)
    print("Start evaluating model.")
    X, y_true = get_matrix_form(features, labels, tile_size)

    y_predicted = model.predict(X)
    predicted_bitmap = np.array(y_predicted)
    predicted_bitmap[0.5 <= predicted_bitmap] = 1
    predicted_bitmap[predicted_bitmap < 0.5] = 0

    visualise_predictions(predicted_bitmap, labels, tile_size, out_path)

    print("Accuracy on test set: {}".format(metrics.accuracy_score(y_true, predicted_bitmap)))
    precision_recall_curve(y_true, y_predicted, out_path)


def visualise_predictions(predictions, labels, tile_size, out_path):
    print("Create .tif result files.")
    predictions = np.reshape(predictions,
                             (len(labels), tile_size, tile_size, 1))
    predictions_transformed = []
    for i, (_, position, path_to_geotiff) in enumerate(labels):
        predictions_transformed.append(
            (predictions[i, :, :, :], position, path_to_geotiff))

    visualise_features(predictions_transformed, tile_size, out_path)


def precision_recall_curve(y_true, y_predicted, out_path):
    print("Calculate precision recall curve.")
    y_true = np.reshape(y_true, (y_true.shape[0] * y_true.shape[1]))
    y_predicted = np.reshape(y_predicted, y_true.shape)
    precision, recall, thresholds = metrics.precision_recall_curve(y_true,
                                                                   y_predicted)
    out_file = os.path.join(out_path, "precision_recall.pickle")
    with open(out_file, "wb") as out:
        pickle.dump({
            "precision": precision,
            "recall": recall,
            "thresholds": thresholds
        }, out)

    out_file = os.path.join(out_path, "precision_recall.png")
    plt.clf()
    plt.plot(recall, precision, label="Precision-Recall curve")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.savefig(out_file)


def get_matrix_form(features, labels, tile_size):
    features = [tile for tile, position, path in features]
    labels = [tile for tile, position, path in labels]
    labels = np.reshape(labels, (len(labels), tile_size * tile_size))
    return np.array(features), np.array(labels)
