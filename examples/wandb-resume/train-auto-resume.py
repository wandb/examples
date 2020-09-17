#!/usr/bin/env python
"""Trains a simple cnn on the fashion mnist dataset.

Demonstrates:
  - Simple integration with Keras - https://docs.wandb.com/library/integrations/keras
  - Automatic resuming - https://docs.wandb.com/library/advanced/resuming

Example:
    Initialize your wandb project
        $ wandb init
    Create keras model and train 4 epochs emulating keyboard interrupt:
        $ python train-auto-resume.py --test_epochs 4
    Resume from previous run training for another 4 epochs:
        $ python train-auto-resume.py --test_epochs 8
    Finish training:
        $ python train-auto-resume.py

"""

import argparse
import sys

from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from keras.models import load_model
import wandb
from wandb.keras import WandbCallback


parser = argparse.ArgumentParser()
parser.add_argument("--test_epochs", type=int, help="Number of epochs to execute")
args = parser.parse_args()


defaults=dict(
    dropout = 0.2,
    hidden_layer_size = 128,
    layer_1_size = 16,
    layer_2_size = 32,
    learn_rate = 0.01,
    decay = 1e-6,
    momentum = 0.9,
    epochs = 10,
    )
run = wandb.init(config=defaults, resume=True)
config = run.config

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
labels=["T-shirt/top","Trouser","Pullover","Dress","Coat",
        "Sandal","Shirt","Sneaker","Bag","Ankle boot"]

img_width=28
img_height=28

X_train = X_train.astype('float32')
X_train /= 255.
X_test = X_test.astype('float32')
X_test /= 255.

X_train = X_train.reshape(X_train.shape[0], img_width, img_height, 1)
X_test = X_test.reshape(X_test.shape[0], img_width, img_height, 1)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Check to see if the run was resumed, if it was, load the best model
if wandb.run.resumed:
    print("Resuming model")
    model = load_model(wandb.restore("model-best.h5").name)
else:
    sgd = SGD(lr=config.learn_rate, decay=config.decay, momentum=config.momentum,
              nesterov=True)
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

# Optionally limit the number of epochs for testing
epochs = args.test_epochs or config.epochs
model.fit(X_train, y_train,  validation_data=(X_test, y_test), epochs=epochs,
    initial_epoch=wandb.run.step,
    callbacks=[WandbCallback(data_type="image", labels=labels)])

# Emulate non-zero exit when limiting the number of epochs for testing
if args.test_epochs:
    raise KeyboardInterrupt
