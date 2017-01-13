import os
import rasterio
import sys
import errno
import pickle
from keras.models import model_from_json
from config import MODELS_DIR


def save_makedirs(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def get_file_name(file_path):
    basename = os.path.basename(file_path)
    # Make sure we don't include the file extension.
    return os.path.splitext(basename)[0]


def save_model_summary(hyperparameters, model, path):
    with open(os.path.join(path, "hyperparameters.txt"), "wb") as out:
        for parameter, value in hyperparameters:
            out.write("{}: {}\n".format(parameter, value))
        stdout = sys.stdout
        sys.stdout = out
        model.summary()
        sys.stdout = stdout


def save_tiles(file_path, tiled_features, tiled_labels):
    print("Store tile data at {}.".format(file_path))
    with open(file_path, "wb") as out:
        pickle.dump({"features": tiled_features, "labels": tiled_labels}, out)


def save_image(file_path, image, source):
    print("Save result at {}.".format(file_path))
    with rasterio.open(
            file_path,
            'w',
            driver='GTiff',
            dtype=rasterio.uint8,
            count=1,
            width=source.width,
            height=source.height,
            transform=source.transform) as dst:
        dst.write(image, indexes=1)


def load_model(model_id):
    model_dir = os.path.join(MODELS_DIR, model_id)
    print("Load model in {}.".format(model_dir))
    model_file = os.path.join(model_dir, "model.json")
    with open(model_file, "r") as f:
        json_file = f.read()
        model = model_from_json(json_file)
    weights_file = os.path.join(model_dir, "weights.hdf5")
    model.load_weights(weights_file)
    return model

def save_model(model, path):
    print("Save trained model to {}.".format(path))
    model_json = model.to_json()
    model_path = os.path.join(path, "model.json")
    with open(model_path, "w") as json_file:
        json_file.write(model_json)
    weights_path = os.path.join(path, "weights.hdf5")
    model.save_weights(weights_path)


