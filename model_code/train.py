from __future__ import absolute_import, division, print_function
import tensorflow as tf
import config
import math
import data_loader
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from res18 import resnet_v1

if __name__ == '__main__':
## Loading data
    train_dataset = data_loader.train_dataset
    valid_dataset = data_loader.valid_dataset
    #train_count = len(data_loader.train_dataset

    """To create the model use the following ->

    DEPTH -> depth should be 6n+2 (eg 20, 32, 44)

    from res18 import resnet_v1
    model = resnet_v1(input_shape = (128, 128, 3), depth = 20 )"""
    # create model
    model = resnet_v1(input_shape = (128, 128, 3), depth = 8)
    # define loss and optimizer
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6)
    model.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.RMSprop(),
              metrics=['accuracy'])
    model.summary()
    model.fit(train_dataset, batch_size=64, epochs=10, validation_data=valid_dataset, validation_freq=1, callbacks=[callback])
    model.save(filepath=config.save_model_dir)
    #reconstructed_model = tf.keras.models.load_model("my_h5_model.h5")

