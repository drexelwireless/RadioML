import tensorflow as tf
from tensorflow import keras
from classification_models.tfkeras import Classifiers

def resnet18(input_shape, nclasses, dropout=0.5, weights='imagenet', **kwargs):
    ResNet18, preprocess = Classifiers.get('resnet18')
    resnet = ResNet18(include_top=False, input_shape=(256,256,3), weights=weights,)

    model = keras.Sequential([
            keras.layers.Input((128,128,3)),
            keras.layers.UpSampling2D(size=(2, 2)),
            resnet,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(dropout),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(nclasses, activation='softmax')
        ], **kwargs)

    return (model, preprocess)

def vt_cnn2(input_shape, nclasses, dropout=0.5, activation='relu', **kwargs):
    """Return the CNN used in "Convolutional Radio Modulation Recognition Networks", O'Shea at al.

    See:
        https://github.com/radioML/examples/blob/master/modulation_recognition/RML2016.10a_VTCNN2_example.ipynb
    """
    model = keras.Sequential([
        keras.layers.Reshape(((1,)+input_shape), input_shape=input_shape),

        keras.layers.ZeroPadding2D((0, 2), data_format='channels_first'),
        keras.layers.Convolution2D(256, (1, 3),
                                padding='valid',
                                activation=activation,
                                name="conv1",
                                kernel_initializer='glorot_uniform',
                                data_format='channels_first'),
        keras.layers.Dropout(dropout),

        keras.layers.ZeroPadding2D((0, 2), data_format='channels_first'),
        keras.layers.Convolution2D(80, (2, 3),
                                padding="valid",
                                activation=activation,
                                name="conv2",
                                kernel_initializer='glorot_uniform',
                                data_format='channels_first'),
        keras.layers.Dropout(dropout),

        keras.layers.Flatten(),
        keras.layers.Dense(256,
                        activation=activation,
                        kernel_initializer='he_normal',
                        name="dense1"),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(nclasses, kernel_initializer='he_normal', name="dense2" ),
        keras.layers.Activation('softmax'),
        keras.layers.Reshape([nclasses])
        ], **kwargs)

    return (model, lambda x : x)
