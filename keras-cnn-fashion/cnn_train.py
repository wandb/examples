#!/usr/bin/env python

"""
Trains a simple cnn on the fashion mnist dataset.
Deigned to show how to do a simple wandb integration with keras.
"""
import argparse
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.optimizers import SGD
from keras.utils import np_utils
# TODO: incorporate Tensorboard
from keras.callbacks import TensorBoard

import wandb
from wandb.keras import WandbCallback

# default config/hyperparameter values
# you can modify these below or via command line
PROJECT_NAME = "fmnist"
MODEL_NAME = "cnn"
BATCH_SIZE = 32
DROPOUT = 0.2
EPOCHS = 25
L1_SIZE = 16
L2_SIZE = 32
HIDDEN_LAYER_SIZE = 128
LEARNING_RATE = 0.01
DECAY = 1e-6
MOMENTUM = 0.9

# dataset config
img_width=28
img_height=28

def train_cnn(args):
  # initialize wandb logging to your project
  wandb.init(project=args.project_name)
  # log all experimental args to wandb
  wandb.config.update(args)

  # load and prepare data
  (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
  labels=["T-shirt/top","Trouser","Pullover","Dress","Coat",
        "Sandal","Shirt","Sneaker","Bag","Ankle boot"]

  X_train = X_train.astype('float32')
  X_train /= 255.
  X_test = X_test.astype('float32')
  X_test /= 255.

  # reshape input data
  X_train = X_train.reshape(X_train.shape[0], img_width, img_height, 1)
  X_test = X_test.reshape(X_test.shape[0], img_width, img_height, 1)

  # one hot encode outputs
  y_train = np_utils.to_categorical(y_train)
  y_test = np_utils.to_categorical(y_test)
  num_classes = y_test.shape[1]

  # build model
  model = Sequential()
  model.add(Conv2D(args.layer_1_size, (5, 5), activation='relu',
                            input_shape=(img_width, img_height,1)))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(args.layer_2_size, (5, 5), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(args.dropout))
  model.add(Flatten())
  model.add(Dense(args.hidden_layer_size, activation='relu'))
  model.add(Dense(num_classes, activation='softmax'))

  sgd = SGD(lr=args.learning_rate, decay=args.decay, momentum=args.momentum,
                            nesterov=True)

  # enable logging for validation examples
  val_generator = ImageDataGenerator()
  model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
  model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=args.epochs,
    callbacks=[WandbCallback(data_type="image", labels=labels, generator=val_generator.flow(X_test, y_test, batch_size=32))])

  # save trained model
  # model.save(args.model_name + ".h5")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "-m",
    "--model_name",
    type=str,
    default=MODEL_NAME,
    help="Name of this model/run (model will be saved to this file)")
  parser.add_argument(
    "-p",
    "--project_name",
    type=str,
    default=PROJECT_NAME,
    help="Main project name")
  parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    default=BATCH_SIZE,
    help="batch_size")
  parser.add_argument(
    "--dropout",
    type=float,
    default=DROPOUT,
    help="dropout before dense layers")
  parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    default=EPOCHS,
    help="number of training epochs (passes through full training data)")
  parser.add_argument(
    "--hidden_layer_size",
    type=int,
    default=HIDDEN_LAYER_SIZE,
    help="hidden layer size")
  parser.add_argument(
    "-l1",
    "--layer_1_size",
    type=int,
    default=L1_SIZE,
    help="layer 1 size")
  parser.add_argument(
    "-l2",
    "--layer_2_size",
    type=int,
    default=L2_SIZE,
    help="layer 2 size")
  parser.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    default=LEARNING_RATE,
    help="learning rate")
  parser.add_argument(
    "--decay",
    type=float,
    default=DECAY,
    help="learning rate decay")
  parser.add_argument(
    "--momentum",
    type=float,
    default=MOMENTUM,
    help="learning rate momentum")
  parser.add_argument(
    "-q",
    "--dry_run",
    action="store_true",
    help="Dry run (do not log to wandb)")  

  args = parser.parse_args()

  # easier testing--don't log to wandb if dry run is set
  if args.dry_run:
    os.environ['WANDB_MODE'] = 'dryrun'

  # create run name
  if not args.model_name:
    print("warning: no run name provided")
  else:
    os.environ['WANDB_DESCRIPTION'] = args.model_name
 
  train_cnn(args)

