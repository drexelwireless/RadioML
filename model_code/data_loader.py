import tensorflow as tf 
import config
import numpy as np
from sklearn.model_selection import train_test_split

train_img = np.load(config.train_img_dir)
train_lbls = np.load(config.train_lbl_dir)

X_train, X_valid, y_train, y_valid = train_test_split(train_img, train_lbls, test_size=0.2, random_state=42)
print("Data Loading finished...")

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
