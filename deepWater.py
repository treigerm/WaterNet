#!/usr/bin/env python

import argparse
import time
import os
from config import SENTINEL_DATASET, TEST_DATASET, OUTPUT_DIR
from preprocessing import preprocess_data
from model import init_model, train_model, evaluate_model

datasets = {
    "sentinel": SENTINEL_DATASET,
    "test": TEST_DATASET
}


def main():
    # TODO: Check if dest needed.
    parser = argparse.ArgumentParser(description="Process satellite images.")
    parser.add_argument("--preprocess_data", dest="preprocess_data", action="store_const",
                        const=True, default=False, help="When selected preprocess data.")
    parser.add_argument("--init-model", dest="init_model", action="store_const",
                        const=True, default=False, help="When selected initialise model.")
    parser.add_argument("--train-model", dest="train_model", action="store_const",
                        const=True, default=False, help="When selected train model.")
    parser.add_argument("--evaluate-model", dest="evaluate_model", action="store_const",
                        const=True, default=False, help="When selected evaluatel model.")
    parser.add_argument("--test-run", action="store_const",
                        const=True, default=False, help="Run on a small test dataset.")
    parser.add_argument("--dataset", default="sentinel", choices=["sentinel"],
                        help="Determine which dataset to use.")
    parser.add_argument("--tile-size", default=64, type=int,
                        help="Choose the tile size.")

    args = parser.parse_args()


    if args.preprocess_data:
        dataset = datasets[args.dataset]
        features_train, features_test, labels_train, labels_test = preprocess_data(args.tile_size,
                                                                                   dataset=dataset)
    else:
        # TODO: Load from cache.
        pass

    if args.test_run:
        dataset = datasets["test"]
        args.dataset = "test"
        features, _, labels, _ = preprocess_data(args.tile_size, dataset=dataset)
        features_train, features_test = features[:100], features[100:120]
        labels_train, labels_test = labels[:100], labels[100:120]

    if args.init_model:
        model = init_model(args.tile_size)
    else:
        # TODO: Load from cache.
        pass

    timestamp = time.strftime("%d_%m_%Y_%H_%M")
    model_id = "{}_{}".format(timestamp, args.dataset)
    if args.train_model:
        model = train_model(model, features_train, labels_train, args.tile_size, model_id)
    else:
        # TODO: Load from cache.
        pass

    if args.evaluate_model:
        model_dir = OUTPUT_DIR + model_id + "/"
        os.makedirs(model_dir)
        evaluate_model(model, features_test, labels_test, args.tile_size, model_dir)

if __name__ == '__main__':
    main()
