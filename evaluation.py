import pickle
import os
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from process_geotiff import visualise_features
from model import get_matrix_form
plt.style.use('ggplot')


def evaluate_model(model, features, labels, tile_size, out_path):
    print('_' * 100)
    print("Start evaluating model.")
    X, y_true = get_matrix_form(features, labels, tile_size)

    y_predicted = model.predict(X)
    predicted_bitmap = np.array(y_predicted)
    predicted_bitmap[0.5 <= predicted_bitmap] = 1
    predicted_bitmap[predicted_bitmap < 0.5] = 0

    visualise_predictions(predicted_bitmap, labels, tile_size, out_path)

    print("Accuracy on test set: {}".format(metrics.accuracy_score(y_true.flatten(), predicted_bitmap.flatten())))
    precision_recall_curve(y_true.flatten(), y_predicted.flatten(), out_path)


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
