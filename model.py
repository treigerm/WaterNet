"""Model used to predict water."""

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, TensorBoard
import rasterio
import pickle
import itertools
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from process_geotiff import read_geotiff, read_bitmap, create_tiles, image_from_tiles, overlay_bitmap
from preprocessing import get_file_name
from config import TENSORBOARD_DIR, WGS84_DIR, MODELS_DIR
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

    checkpointer = None
    if checkpoints:
        checkpoints_path = MODELS_DIR + model_id + "/checkpoints/"
        # TODO: Create path.
        checkpointer = ModelCheckpoint(checkpoints_path)

    tensorboarder = None
    if tensorboard:
        tensorboarder = TensorBoard(log_dir=TENSORBOARD_DIR)

    callbacks = [c for c in [checkpointer, tensorboarder] if c]
    print("Start training.")
    model.fit(X, y, nb_epoch=nb_epoch, callbacks=callbacks, validation_split=0.1)
    return model


def init_model(tile_size,
               num_channels=3,
               filter_size=12,
               stride=4,
               nb_filters=64):
    model = Sequential()

    model.add(
        Convolution2D(
            nb_filters,
            filter_size,
            filter_size,
            subsample=(stride, stride),
            input_shape=(tile_size, tile_size, num_channels)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(tile_size * tile_size, activation='sigmoid'))

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

    precision_recall_curve(y_true, y_predicted, out_path)


def visualise_predictions(predictions, labels, tile_size, out_path):
    print("Create .tif result files.")
    predictions = np.reshape(predictions,
                             (len(labels), tile_size, tile_size, 1))
    predictions_transformed = []
    for i, (_, position, path_to_geotiff) in enumerate(labels):
        predictions_transformed.append(
            (predictions[i, :, :, :], position, path_to_geotiff))

    get_path = lambda x: x[2]
    sorted_by_path = sorted(predictions_transformed, key=get_path)
    for path, predictions in itertools.groupby(sorted_by_path, get_path):
        satellite_img_name = get_file_name(path)
        path_wgs84 = WGS84_DIR + satellite_img_name + "_wgs84.tif"
        raster_dataset = rasterio.open(path_wgs84)
        bitmap_shape = (raster_dataset.shape[0], raster_dataset.shape[1], 1)
        bitmap = image_from_tiles(predictions, tile_size, bitmap_shape)
        bitmap = np.reshape(bitmap, (bitmap.shape[0], bitmap.shape[1]))
        satellite_img_name = get_file_name(path)
        overlay_bitmap(bitmap, raster_dataset,
                       out_path + satellite_img_name + ".tif")


def precision_recall_curve(y_true, y_predicted, out_path):
    print("Calculate precision recall curve.")
    y_true = np.reshape(y_true, (y_true.shape[0] * y_true.shape[1]))
    y_predicted = np.reshape(y_predicted, y_true.shape)
    precision, recall, thresholds = metrics.precision_recall_curve(y_true,
                                                                   y_predicted)
    with open(out_path + "precision_recall.pickle", "wb") as out:
        pickle.dump({
            "precision": precision,
            "recall": recall,
            "thresholds": thresholds
        }, out)

    plt.clf()
    plt.plot(recall, precision, lw=2, label="Precision-Recall curve")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.savefig(out_path + "precision_recall.png")


def get_matrix_form(features, labels, tile_size):
    features = [tile for tile, position, path in features]
    labels = [tile for tile, position, path in labels]
    labels = np.reshape(labels, (len(labels), tile_size * tile_size))
    return np.array(features), np.array(labels)
