import tensorflow as tf 
import config
import numpy as np
from sklearn.model_selection import train_test_split
print("Adeeb")

train_img = np.load(config.train_img_dir)
train_lbls = np.load(config.train_lbl_dir)
print("Abbas")
X_train, X_valid, y_train, y_valid = train_test_split(train_img, train_lbls, test_size=0.2, random_state=42)
print("Done")
print(len(list(X_train.values)))


# train_dataset = tf.data.Dataset.from_tensor_slices((train_img, train_lbls)).shuffle(50)
# for testing it will use the eval file