"""Evaluate a models performance."""

import matplotlib
# Use 'Agg' backend to be able to use matplotlib in Docker container.
# See http://stackoverflow.com/questions/4931376/generating-matplotlib-graphs-without-a-running-x-server.
matplotlib.use('Agg')

import pickle
import os
import numpy as np
from sklearn import metrics
from geo_util import visualise_results
from model import get_matrix_form

import matplotlib.pyplot as plt
plt.style.use('ggplot')


def evaluate_model(model, features, labels, tile_size, out_path, out_format="GeoTIFF"):
    """Calculate several metrics for the model and create a visualisation of the test dataset."""

    print('_' * 100)
    print("Start evaluating model.")

    X, y_true = get_matrix_form(features, labels, tile_size)

    y_predicted = model.predict(X)
    predicted_bitmap = np.array(y_predicted)

    # Since the model only outputs probabilites for each pixel we have
    # to transform them into 0s and 1s. For the sake of simplicity we
    # simply use a cut off value of 0.5.
    predicted_bitmap[0.5 <= predicted_bitmap] = 1
    predicted_bitmap[predicted_bitmap < 0.5] = 0

    false_positives = get_false_positives(predicted_bitmap, y_true)
    visualise_predictions(predicted_bitmap, labels, false_positives, tile_size, out_path, out_format=out_format)

    # We have to flatten our predictions and labels since by default the metrics are calculated by
    # comparing the elements in the list of labels and predictions elemtwise. So if we would not flatten
    # our results we would only get a true positive if we would predict every pixel in an entire tile right.
    # But we obviously only care about each pixel individually.
    y_true = y_true.flatten()
    y_predicted = y_predicted.flatten()
    predicted_bitmap = predicted_bitmap.flatten()

    print("Accuracy on test set: {}".format(metrics.accuracy_score(y_true, predicted_bitmap)))
    print("Precision on test set: {}".format(metrics.precision_score(y_true, predicted_bitmap)))
    print("Recall on test set: {}".format(metrics.recall_score(y_true, predicted_bitmap)))
    precision_recall_curve(y_true, y_predicted, out_path)


def visualise_predictions(predictions, labels, false_positives, tile_size, out_path, out_format="GeoTIFF"):
    """Create a new GeoTIFF image which overlays the predictions of the model."""

    print("Create {} result files.".format(out_format))
    predictions = np.reshape(predictions,
                             (len(labels), tile_size, tile_size, 1))
    false_positives = np.reshape(false_positives,
                             (len(labels), tile_size, tile_size, 1))

    results = []
    # We want to overlay the predictions and false positives on a GeoTIFF but we don't
    # have any information about the source image and the position in the source for each
    # tile in the predictions and false postives. We get this information from the labels.
    for i, (_, position, path_to_geotiff) in enumerate(labels):
        prediction_tile = predictions[i, :, :, :]
        false_positivle_tile = false_positives[i, :, :, :]
        label_tile = labels[i][0]
        results.append(
            ((prediction_tile, label_tile, false_positivle_tile), position, path_to_geotiff))

    visualise_results(results, tile_size, out_path, out_format=out_format)


def precision_recall_curve(y_true, y_predicted, out_path):
    """Create a PNG with the precision-recall curve for our predictions."""

    print("Calculate precision recall curve.")
    precision, recall, thresholds = metrics.precision_recall_curve(y_true,
                                                                   y_predicted)

    # Save the raw precision and recall results to a pickle since we might want
    # to analyse them later.
    out_file = os.path.join(out_path, "precision_recall.pickle")
    with open(out_file, "wb") as out:
        pickle.dump({
            "precision": precision,
            "recall": recall,
            "thresholds": thresholds
        }, out)

    # Create the precision-recall curve.
    out_file = os.path.join(out_path, "precision_recall.png")
    plt.clf()
    plt.plot(recall, precision, label="Precision-Recall curve")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.savefig(out_file)

def get_false_positives(predictions, labels):
    """Get false positives for the given predictions and labels."""

    FP = np.logical_and(predictions == 1, labels == 0)
    false_positives = np.copy(predictions)
    false_positives[FP] = 1
    false_positives[np.logical_not(FP)] = 0
    return false_positives
