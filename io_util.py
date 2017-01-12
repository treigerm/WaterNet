import os
import sys
import errno


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


def save_model(model, path):
    print("Save trained model to {}.".format(path))
    model_json = model.to_json()
    model_path = os.path.join(path, "trained_model.json")
    with open(model_path, "w") as json_file:
        json_file.write(model_json)
    weights_path = os.path.join(path, "trained_model_weights.hdf5")
    model.save_weights(weights_path)


class Logger(object):
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(os.path.join(path, "logfile.log"), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
