"""Model used to predict water."""

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from process_geotiff import read_geotiff, read_bitmap, create_tiles, image_from_tiles, overlay_bitmap
from transform_shapefile import create_image, muenster_img, muenster_water, DATA_DIR, CACHE_DIR
plt.style.use('ggplot')

MODEL_CACHE = DATA_DIR + "working/logs/"


def normalize_input(features):
    features = features.astype(np.float32)
    return np.multiply(features, 1.0 / 255.0)


def train_model(model, features, labels, tile_size):
    features = [tile for tile, position in features]
    labels = [tile for tile, position in labels]
    labels = np.reshape(labels, (len(labels), tile_size * tile_size))
    features = normalize_input(np.array(features))
    print("Start training.")
    model.fit(features, labels, validation_set=0.1)  # TODO: Hyperparameters.
    return model


def init_model(tile_size):
    # TODO: Declare 3 a variable as num_channels.
    # From DeepOSM.
    net = tflearn.input_data(shape=[None, tile_size, tile_size, 3])
    net = conv_2d(net, 64, 12, strides=4, activation='relu')
    net = max_pool_2d(net, 3)

    # TODO: Switch to sigmoid?
    softmax = tflearn.fully_connected(
        net, tile_size * tile_size, activation='softmax')

    momentum = tflearn.optimizers.Momentum(
        learning_rate=0.005, momentum=0.9,
        lr_decay=0.002, name='Momentum'
    )

    net = tflearn.regression(softmax, optimizer=momentum,
                             loss='categorical_crossentropy')

    return tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir=MODEL_CACHE)


def evaluate_model(model, features, labels, tile_size, raster_dataset, out_path):
    # TODO: Raster dataset not as argument.
    features = [tile for tile, position in features]  # TODO: Make function.
    predictions = model.predict(features)
    predicted_bitmap = np.array(predictions)
    predicted_bitmap[0.5 <= predicted_bitmap] = 1
    predicted_bitmap[predicted_bitmap < 0.5] = 0

    visualise_predictions(predicted_bitmap, labels,
                          tile_size, raster_dataset, out_path)

    labels = [tile for tile, position in labels]
    labels = np.reshape(labels, (len(labels) * tile_size * tile_size))
    predictions = np.reshape(predictions, (len(labels)))
    precision, recall, thresholds = metrics.precision_recall_curve(
        labels, predictions)
    # TODO: Store in pickle.

    plt.clf()
    plt.plot(recall, precision, lw=2, label="Precision-Recall curve")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.savefig(CACHE_DIR + "precision_recall.png")


def visualise_predictions(predictions, labels, tile_size, raster_dataset, out_path):
    predictions = np.reshape(
        predictions, (len(labels), tile_size, tile_size, 1))
    predictions_transformed = []
    for i, (_, position) in enumerate(labels):
        predictions_transformed.append((predictions[i, :, :, :], position))
    # TODO: Create parameters.
    bitmap = image_from_tiles(predictions_transformed,
                              tile_size, (5490, 5490, 1))
    bitmap = np.reshape(bitmap, (bitmap.shape[0], bitmap.shape[1]))
    overlay_bitmap(bitmap, raster_dataset, out_path)

if __name__ == '__main__':
    tile_size = 64
    tile_overlap = 0
    geotiffs = [(muenster_img, [muenster_water])]
    features = []
    labels = []
    for geotiff, shapefiles in geotiffs:
        dataset, bands = read_geotiff(geotiff)
        water_img = create_image(dataset, shapefiles, geotiff)
        water_img[water_img == 255] = 1
        tiled_bands = create_tiles(bands, tile_size)
        tiled_bitmap = create_tiles(water_img, tile_size)
        features += tiled_bands
        labels += tiled_bitmap

    model = init_model(tile_size)
    model = train_model(model, features[:100], labels[:100], tile_size)
    evaluate_model(model, features[100:120], labels[
                   100:120], tile_size, dataset, CACHE_DIR + "result.tif")
