#!/usr/bin/env python

import argparse
import time
import os
import sys
from config import SENTINEL_DATASET, DEBUG_DATASET, OUTPUT_DIR, TRAIN_DATA_DIR
from preprocessing import preprocess_data, visualise_features
from model import init_model, train_model, evaluate_model
from io_util import save_makedirs, Logger

datasets = {
    "sentinel": SENTINEL_DATASET,
    "test": DEBUG_DATASET
}


def main():
    # TODO: Check if dest needed.
    parser = argparse.ArgumentParser(description="Process satellite images.")
    parser.add_argument("-p, --preprocess_data", dest="preprocess_data", action="store_const",
                        const=True, default=False, help="When selected preprocess data.")
    parser.add_argument("-i, --init-model", dest="init_model", action="store_const",
                        const=True, default=False, help="When selected initialise model.")
    parser.add_argument("-t, --train-model", dest="train_model", action="store_const",
                        const=True, default=False, help="When selected train model.")
    parser.add_argument("-e, --evaluate-model", dest="evaluate_model", action="store_const",
                        const=True, default=False, help="When selected evaluatel model.")
    parser.add_argument("-d, --debug", dest="debug", action="store_const",
                        const=True, default=False, help="Run on a small test dataset.")
    parser.add_argument("--dataset", default="sentinel", choices=["sentinel"],
                        help="Determine which dataset to use.")
    parser.add_argument("--tile-size", default=64, type=int,
                        help="Choose the tile size.")
    parser.add_argument("--epochs", default=10, type=int,
                        help="Number of training epochs.")

    args = parser.parse_args()

    timestamp = time.strftime("%d_%m_%Y_%H%M")
    model_id = "{}_{}".format(timestamp, args.dataset)
    model_dir = os.path.join(OUTPUT_DIR, model_id)
    save_makedirs(model_dir)

    sys.stdout = Logger(model_dir)

    if args.debug:
        dataset = datasets["test"]
        args.dataset = "test"
        features, _, labels, _ = preprocess_data(args.tile_size, dataset=dataset)
        features_train, features_test = features[:100], features[100:120]
        labels_train, labels_test = labels[:100], labels[100:120]
    elif args.preprocess_data:
        dataset = datasets[args.dataset]
        features_train, features_test, labels_train, labels_test = preprocess_data(args.tile_size,
                                                                                   dataset=dataset)
        # TODO: Make option to visualise.
        #visualise_features(labels_train, args.tile_size, TRAIN_DATA_DIR)
        #visualise_features(labels_test, args.tile_size, TRAIN_DATA_DIR)
    else:
        pass


    if args.init_model:
        model = init_model(args.tile_size)
    else:
        # TODO: Load from cache.
        pass

    if args.train_model:
        model = train_model(model, features_train, labels_train, args.tile_size, model_id, nb_epoch=args.epochs)
    else:
        # TODO: Load from cache.
        pass

    if args.evaluate_model:
        evaluate_model(model, features_test, labels_test, args.tile_size, model_dir)

if __name__ == '__main__':
    main()
