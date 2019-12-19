#!/usr/bin/env python

"""
Trains a simple cnn on the fashion mnist dataset.
Deigned to show how to do a simple wandb integration with keras.
"""

from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.callbacks import TensorBoard

import wandb
from wandb.keras import WandbCallback


defaults=dict(
    dropout = 0.2,
    hidden_layer_size = 128,
    layer_1_size = 16,
    layer_2_size = 32,
    learn_rate = 0.01,
    decay = 1e-6,
    momentum = 0.9,
    epochs = 27,
    )

wandb.init(config=defaults)
config = wandb.config

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
labels=["T-shirt/top","Trouser","Pullover","Dress","Coat",
        "Sandal","Shirt","Sneaker","Bag","Ankle boot"]

img_width=28
img_height=28

X_train = X_train.astype('float32')
X_train /= 255.
X_test = X_test.astype('float32')
X_test /= 255.

#reshape input data
X_train = X_train.reshape(X_train.shape[0], img_width, img_height, 1)
X_test = X_test.reshape(X_test.shape[0], img_width, img_height, 1)

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

sgd = SGD(lr=config.learn_rate, decay=config.decay, momentum=config.momentum,
                            nesterov=True)

# build model
model = Sequential()
model.add(Conv2D(config.layer_1_size, (5, 5), activation='relu',
                            input_shape=(img_width, img_height,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(config.layer_2_size, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(config.dropout))
model.add(Flatten())
model.add(Dense(config.hidden_layer_size, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X_train, y_train,  validation_data=(X_test, y_test), epochs=config.epochs,
    callbacks=[WandbCallback(data_type="image", labels=labels)])

model.save("cnn.h5")
