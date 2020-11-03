import tensorflow as tf 
import config
import numpy as np
from sklearn.model_selection import train_test_split
#import tensorflow
train_img = np.load(config.train_img_dir)
train_lbls = np.load(config.train_lbl_dir)

print(train_img.shape)
#print(train_img.lbls)
#train_lbls.reshape((len(train_lbls), 1, 11))

X_train, X_valid, y_train, y_valid = train_test_split(train_img, train_lbls, test_size=0.2, random_state=42)
print("Data Loading finished...")


## One hot encoding! 

#y_train = tensorflow.keras.utils.to_categorical(y_train, 11)
#y_valid = tensorflow.keras.utils.to_categorical(y_valid, 11)

#print(y_train)
#print(y_train.shape)
print(train_lbls.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
