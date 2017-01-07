#!/usr/bin/env python

import argparse
from config import SENTINEL_DATASET, OUTPUT_IMGS_DIR
from transform_shapefile import preprocess_data
from model import init_model, train_model, evaluate_model

datasets = {
    "sentinel": SENTINEL_DATASET
}


def main():
    # TODO: Check if dest needed.
    parser = argparse.ArgumentParser(description="Process satellite images.")
    parser.add_argument("--preprocess_data", dest="preprocess_data", action="store_const",
                        const=True, default=False, help="When selected preprocess data.")
    parser.add_argument("--init_model", dest="init_model", action="store_const",
                        const=True, default=False, help="When selected initialise model.")
    parser.add_argument("--train_model", dest="train_model", action="store_const",
                        const=True, default=False, help="When selected train model.")
    parser.add_argument("--evaluate_model", dest="evaluate_model", action="store_const",
                        const=True, default=False, help="When selected evaluatel model.")
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

    if args.initialise_model:
        model = init_model(tile_size)
    else:
        # TODO: Load from cache.
        pass

    if args.train_model:
        model = train_model(model, features_train, labels_train, tile_size)
    else:
        # TODO: Load from cache.
        pass

    if args.evaluate_model:
        out_path = OUTPUT_IMGS_DIR
        evaluate_model(model, features_test, labels_test, tile_size, out_path)
        pass

if __name__ == '__main__':
    main()
