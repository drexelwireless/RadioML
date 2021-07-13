import datetime
import logging
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np

import tensorflow as tf
import tensorflow.keras as keras

def set_seed(seed : int):
    """Set random seed.

    This initalizes the Tensorflow and Numpy random seeds to aid reproducibility.

    Args:
        seed: The random seed.
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)

def split_training(arr : np.ndarray, train : float, validate : float = 0.0) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Split data into training, validation, and test sets.

    Args:
        arr: NumPy array to split
        train: Fraction of data to use for training
        validate: Fraction of data to use for validation

    Returns:
        A tuple of (train, validate, test)
    """
    n = len(arr)
    idx = np.arange(0, n, dtype=int)
    np.random.shuffle(idx)

    train_idx = int(train*n)
    val_idx = train_idx + int(validate*n)

    return arr[idx[:train_idx]], arr[idx[train_idx:val_idx]], arr[idx[val_idx:]]

class Trainer:
    def __init__(self,
                 model_dir,
                 dataset_name,
                 model_name,
                 seed,
                 patience=15):
        self.model_dir = model_dir
        """Directory containing saved models and checkpoints"""

        self.dataset_name = dataset_name
        """Name of data set to use"""

        self.model_name = model_name
        """Name of model to use"""

        self.seed = seed
        """Seed used for training"""

        self.patience = patience
        """Number of epochs with no improvement after which training will be stopped."""

        self.model = None
        """Model to train"""

    @property
    def checkpoint_path(self):
        return os.path.join(self.model_dir,
                            "checkpoint",
                            self.dataset_name,
                            self.model_name,
                            str(self.seed))

    @property
    def history_path(self):
        return os.path.join(self.model_dir,
                            "history",
                            self.dataset_name,
                            self.model_name,
                            str(self.seed) + '.pkl')

    @property
    def weights_path(self):
        return os.path.join(self.model_dir,
                            "weights",
                            self.dataset_name,
                            self.model_name,
                            str(self.seed))

    @property
    def predictions_path(self):
        return os.path.join(self.model_dir,
                            "predictions",
                            self.dataset_name,
                            self.model_name,
                            str(self.seed) + '.npy')

    def tensorboard_path(self):
        return os.path.join(self.model_dir,
                            "tensorboard",
                            datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    def callbacks(self):
        """Construct Keras training callbacks"""
        tensorboard_cb = keras.callbacks.TensorBoard(log_dir=self.tensorboard_path())

        checkpoint_cb = keras.callbacks.ModelCheckpoint(os.path.join(self.checkpoint_path, "{epoch}"),
                                                        monitor='val_loss',
                                                        verbose=0,
                                                        save_best_only=True,
                                                        mode='auto')

        stopping_cb = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=self.patience,
                                                    verbose=0,
                                                    mode='auto',
                                                    restore_best_weights=True)

        return [ tensorboard_cb
               , checkpoint_cb
               , stopping_cb
               ]

    def load_weights(self):
        self.model.load_weights(self.weights_path)

    def save_weights(self):
        os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
        path_ = os.path.join(self.weights_path, "model.h5")
        self.model.save(path_)
        ##save as the onnx format
        self.model.save_weights(self.weights_path)

    def load_history(self):
        with open(self.history_path, 'rb') as f:
            return pickle.load(f)

    def save_history(self, history):
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)

        with open(self.history_path, 'wb') as f:
            pickle.dump(history, f)

    def load_predictions(self):
        return np.load(self.predictions_path)

    def save_predictions(self, data):
        os.makedirs(os.path.dirname(self.predictions_path), exist_ok=True)
        np.save(self.predictions_path, data)

    def load_model(self):
        if os.path.exists(self.weights_path):
            logging.info("Restoring model from %s", self.weights_path)
            return keras.models.load_model(self.weights_path)

        if self.checkpoint_path is not None and os.path.exists(self.checkpoint_path):
            checkpoints = [os.path.join(self.checkpoint_path, name) for name in os.listdir(self.checkpoint_path)]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=os.path.getctime)
                logging.info("Restoring model from %s", latest_checkpoint)
                return keras.models.load_model(latest_checkpoint)

        return None

    def compile(self):
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def fit(self, x=None, y=None, verbose=1, epochs=100, **kwargs):
        history = self.model.fit(x=x,
                                 y=y,
                                 verbose=verbose,
                                 epochs=epochs,
                                 callbacks=self.callbacks(),
                                 **kwargs)

        # Save weights
        #self.model.save(self.weights_path)
        self.save_weights()

        # Save fit history
        if os.path.exists(self.history_path):
            complete_history = self.load_history()
        else:
            complete_history = []

        complete_history.append(history.history)

        self.save_history(complete_history)

        return history
