import tensorflow as tf 
import config
import numpy as np

train_img = np.load(config.train_img_dir)
train_lbls = np.load(config.train_lbl_dir)

train_dataset = tf.data.Dataset.from_tensor_slices((train_img, train_lbls))


