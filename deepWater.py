#!/usr/bin/env python

import argparse
import time
import os
import sys
from config import SENTINEL_DATASET, DEBUG_DATASET, OUTPUT_DIR, TRAIN_DATA_DIR
from preprocessing import preprocess_data
from model import init_model, train_model
from evaluation import evaluate_model
from io_util import save_makedirs, save_model_summary, load_model
from geo_util import visualise_features

datasets = {"sentinel": SENTINEL_DATASET, "debug": DEBUG_DATASET}


def create_parser():
    parser = argparse.ArgumentParser(description="Process satellite images.")
    parser.add_argument(
        "-p, --preprocess-data",
        dest="preprocess_data",
        action="store_const",
        const=True,
        default=False,
        help="When selected preprocess data.")
    parser.add_argument(
        "-i, --init-model",
        dest="init_model",
        action="store_const",
        const=True,
        default=False,
        help="When selected initialise model.")
    parser.add_argument(
        "-t, --train-model",
        dest="train_model",
        action="store_const",
        const=True,
        default=False,
        help="When selected train model.")
    parser.add_argument(
        "-e, --evaluate-model",
        dest="evaluate_model",
        action="store_const",
        const=True,
        default=False,
        help="When selected evaluatel model.")
    parser.add_argument(
        "-d, --debug",
        dest="debug",
        action="store_const",
        const=True,
        default=False,
        help="Run on a small test dataset.")
    parser.add_argument(
        "-a, --architecture",
        dest="architecture",
        default="one_layer",
        choices=["one_layer", "two_layer"],
        help="Neural net architecture.")
    parser.add_argument(
        "-v, --visualise",
        dest="visualise",
        default=False,
        action="store_const",
        const=True,
        help="Visualise labels.")
    parser.add_argument(
        "-T, --tensorboard",
        dest="tensorboard",
        default=False,
        action="store_const",
        const=True,
        help="Store tensorboard data while training.")
    parser.add_argument(
        "-C, --checkpoints",
        dest="checkpoints",
        default=False,
        action="store_const",
        const=True,
        help="Create checkpoints while training.")
    parser.add_argument(
        "--dataset",
        default="sentinel",
        choices=["sentinel"],
        help="Determine which dataset to use.")
    parser.add_argument(
        "--tile-size", default=64, type=int, help="Choose the tile size.")
    parser.add_argument(
        "--epochs", default=10, type=int, help="Number of training epochs.")
    parser.add_argument(
        "--model-id",
        default=None,
        type=str,
        help="Model that should be used.")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.debug:
        dataset = datasets["debug"]
        args.dataset = "debug"
        features, _, labels, _ = preprocess_data(
            args.tile_size, dataset=dataset)
        features_train, features_test = features[:100], features[100:120]
        labels_train, labels_test = labels[:100], labels[100:120]
    elif args.train_model or args.evaluate_model or args.preprocess_data:
        dataset = datasets[args.dataset]
        load_from_cache = not args.preprocess_data
        try:
            features_train, features_test, labels_train, labels_test = preprocess_data(
                args.tile_size, dataset=dataset, only_cache=load_from_cache)
        except IOError:
            print("Cache file does not exist. Please run again with -p flag.")

        if args.visualise:
            out_dir = os.path.join(TRAIN_DATA_DIR, "labels_images")
            visualise_features(labels_train, args.tile_size, out_dir)
            visualise_features(labels_test, args.tile_size, out_dir)

    if not args.model_id:
        timestamp = time.strftime("%d_%m_%Y_%H%M")
        model_id = "{}_{}_{}".format(timestamp, args.dataset, args.architecture)
    else:
        model_id = args.model_id

    model_dir = os.path.join(OUTPUT_DIR, model_id)
    save_makedirs(model_dir)

    if args.init_model:
        hyperparameters = [
            ("architecture", args.architecture),
            ("nb_filters_1", 64),
            ("filter_size_1", 12),
            ("stride_1", (4, 4)),
            ("pool_size_1", (3, 3)),
            ("nb_filters_2", 128),
            ("filter_size_2", 4),
            ("stride_2", (1, 1)),
            ("learning_rate", 0.005),
            ("momentum", 0.9),
            ("decay", 0.002)
        ]
        model = init_model(args.tile_size, model_id, **dict(hyperparameters))
        save_model_summary(hyperparameters, model, model_dir)
    elif args.train_model or args.evaluate_model:
        model = load_model(model_id)

    if args.train_model:
        model = train_model(
            model,
            features_train,
            labels_train,
            args.tile_size,
            model_id,
            nb_epoch=args.epochs,
            checkpoints=args.checkpoints,
            tensorboard=args.tensorboard)

    if args.evaluate_model:
        evaluate_model(model, features_test, labels_test, args.tile_size,
                       model_dir)


if __name__ == '__main__':
    main()
