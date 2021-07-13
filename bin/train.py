#!/usr/bin/env python3
import argparse
import datetime
import logging
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

import modrec.models
import modrec.preprocessing
import modrec.radioml
import modrec.training

def to_tfdata(classes, data):
    return tf.data.Dataset.from_tensor_slices((data['iq_data'], tf.one_hot(data['ms'], depth=len(classes))))

def get_model(model_name, input_shape, nclasses):
    n = input_shape[-1]

    if model_name == 'vtcnn2':
        return modrec.models.vt_cnn2(input_shape, nclasses)
    elif model_name == 'resnet18-outer':
        model, preprocess = modrec.models.resnet18((n, n, 3), nclasses, dropout=0.5)

        def f(x):
            return preprocess(256*modrec.preprocessing.preprocess_outer(x))

        return model, f
    elif model_name == 'resnet18-gasf':
        model, preprocess = modrec.models.resnet18((n, n, 3), nclasses, dropout=0.5)

        def f(x):
            return preprocess(256*modrec.preprocessing.preprocess_gasf(x))

        return model, f
    elif model_name == 'resnet18-gadf':
        model, preprocess = modrec.models.resnet18((n, n, 3), nclasses, dropout=0.5)

        def f(x):
            return preprocess(256*modrec.preprocessing.preprocess_gadf(x))

        return model, f
    else:
        raise ValueError("Unknown model %s", model_name)

def main():
    parser = argparse.ArgumentParser(description='Train on RadioML 2016 data.')
    parser.add_argument('-d', '--debug', action='store_const', const=logging.DEBUG,
                        dest='loglevel',
                        default=logging.WARNING,
                        help='print debugging information')
    parser.add_argument('-v', '--verbose', action='store_const', const=logging.INFO,
                        dest='loglevel',
                        help='be verbose')
    parser.add_argument('--seed', type=int,
                        default=2016,
                        help='set random seed')
    parser.add_argument('--batch-size', type=int,
                        default=1024,
                        help='batch size')
    parser.add_argument('--train', action='store_true',
                        default=False,
                        help='train model')
    parser.add_argument('--predict', action='store_true',
                        default=False,
                        help='use model for prediction')
    parser.add_argument('--model',
                        default='vtcnn2',
                        choices=[ 'vtcnn2'
                                , 'resnet18-outer'
                                , 'resnet18-gasf'
                                , 'resnet18-gadf'
                                ],
                        help='model to use')
    parser.add_argument('input', help='RadioML data in standard HDF5 format')

    # Parse arguments
    try:
        args = parser.parse_args()
    except SystemExit as ex:
        return ex.code

    # Set up logging
    logging.basicConfig(format='%(asctime)s:%(name)s:%(levelname)s:%(message)s',
                        level=args.loglevel)

    # Allow memory growth on GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    logging.info(physical_devices)
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    # Set random seeds
    modrec.training.set_seed(args.seed)

    # Load RadioML 2016 data
    classes, data = modrec.radioml.load_numpy(args.input)

    dataset_name = os.path.splitext(os.path.basename(args.input))[0]

    input_shape = data['iq_data'].shape[1:]

    # Create model
    trainer = modrec.training.Trainer('models', dataset_name, args.model, args.seed)

    model, preprocess = get_model(args.model, input_shape, len(classes))

    trainer.model = model

    if args.predict:
        trainer.load_weights()

    trainer.compile()
    logging.info(trainer.model.summary())

    # Split into training, validate, and test sets
    train, validate, test = modrec.training.split_training(data, 0.5, 0.5)

    train_dataset = to_tfdata(classes, train)
    validate_dataset = to_tfdata(classes, validate)
    test_dataset = to_tfdata(classes, test)

    # Map preprocessing function over data
    def f(x, lbl):
        return (preprocess(x), lbl)

    train_dataset = train_dataset.map(f).batch(args.batch_size).prefetch(1)
    validate_dataset = validate_dataset.map(f).batch(args.batch_size).prefetch(1)
    test_dataset = test_dataset.map(f).batch(args.batch_size).prefetch(1)

    if args.train:
        trainer.fit(train_dataset, validation_data=validate_dataset)

    if args.predict:
        predictions = trainer.model.predict(validate_dataset, verbose=2)

        trainer.save_predictions(predictions)

    score = trainer.model.evaluate(validate_dataset,
                                   verbose=0)
    print(score)

if __name__ == "__main__":
    main()
