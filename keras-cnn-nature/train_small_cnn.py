import argparse
import os

from keras import backend as K
from keras.callbacks import Callback
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

import wandb
from wandb.keras import WandbCallback

# training config
#--------------------------
img_width, img_height = 299, 299

def build_model(optimizer, dropout, num_classes):
  """ Construct a simple categorical CNN following the Keras tutorial """
  if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
  else:
    input_shape = (img_width, img_height, 3)

  model = Sequential()
  model.add(Conv2D(16, (3, 3), input_shape=input_shape))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(32, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(32, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(64, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(128, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Flatten())
  model.add(Dense(128))
  model.add(Activation('relu'))
  model.add(Dropout(dropout))
  model.add(Dense(num_classes))
  model.add(Activation('softmax'))

  model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
  return model

def log_model_params(model, wandb_config, args):
  """ Extract params of interest about the model (e.g. number of different layer types).
      Log these and any experiment-level settings to wandb """
  num_conv_layers = 0
  num_fc_layers = 0
  for l in model.layers:
    layer_type = l.get_config()["name"].split("_")[0]
    if layer_type == "conv2d":
      num_conv_layers += 1
    elif layer_type == "dense":
      num_fc_layers += 1

  wandb_config.update({
    "epochs" : args.epochs,
    "batch_size" : args.batch_size,
    "n_conv_layers" : num_conv_layers,
    "n_fc_layers" : num_fc_layers,
    "img_dim" : img_width,
    "num_classes" : args.num_classes,
    "n_train" : args.num_train,
    "n_valid" : args.num_valid,
    "optimizer" : args.optimizer,
    "dropout" : args.dropout
  })

def run_experiment(args):
  """ Build model and data generators; run training"""
  wandb.init(project="keras-cnn-nature")

  model = build_model(args.optimizer, args.dropout, args.num_classes)
  # log all values of interest to wandb
  log_model_params(model, wandb.config, args)

  # training and test data generated as in Keras tutorial
  train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
  test_datagen = ImageDataGenerator(rescale=1. / 255)

  train_generator = train_datagen.flow_from_directory(
    args.train_data,
    target_size = (img_width, img_height),
    batch_size = args.batch_size,
    class_mode = 'categorical',
    follow_links = True)

  validation_generator = test_datagen.flow_from_directory(
    args.val_data,
    target_size = (img_width, img_height),
    batch_size = args.batch_size,
    class_mode = 'categorical',
    follow_links = True)

  callbacks = [WandbCallback()]

  # core training method  
  model.fit_generator(
    train_generator,
    steps_per_epoch = args.num_train // args.batch_size,
    epochs = args.epochs,
    validation_data = validation_generator,
    callbacks = callbacks,
    validation_steps = args.num_valid // args.batch_size)

  # save the model weights
  save_model_filename = args.model_name + ".h5"
  model.save_weights(save_model_filename)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # Strongly recommended args
  #----------------------------
  parser.add_argument(
    "-m",
    "--model_name",
    type=str,
    default="",
    help="Name of this model/run (model will be saved to this file)")
  
  # Optional args
  #----------------------------
  parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    default=32,
    help="Batch size")
  parser.add_argument(
    "-c",
    "--num_classes",
    type=int,
    default=10,
    help="Number of classes to predict")
  parser.add_argument(
    "-d",
    "--dropout",
    type=float,
    default=0.3,
    help="Dropout before the last fc layer") 
  parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    default=50,
    help="Number of training epochs")
  parser.add_argument(
    "-nt",
    "--num_train",
    type=int,
    default=5000,
    help="Number of training examples")
  parser.add_argument(
    "-nv",
    "--num_valid",
    type=int,
    default=800,
    help="Number of validation examples")
  parser.add_argument(
    "-o",
    "--optimizer",
    type=str,
    default="adam",
    help="Learning optimizer")
  parser.add_argument(
    "-t",
    "--train_data",
    type=str,
    default="~/nature_12K/train",
    help="Absolute path to training data")
  parser.add_argument(
    "-v",
    "--val_data",
    type=str,
    default="~/nature_12K/val",
    help="Absolute path to validation data")  
  parser.add_argument(
    "-q",
    "--dry_run",
    action="store_true",
    help="Dry run (if set, do not log to wandb)")
 
  args = parser.parse_args()

  # easier iteration/testing--don't log to wandb if dry run is set
  if args.dry_run:
    os.environ['WANDB_MODE'] = 'dryrun'

  # strongly recommend a descriptive run name
  if not args.model_name:
    print "warning: no run name provided"
    args.model_name = "model"
  else:
    os.environ['WANDB_DESCRIPTION'] = args.model_name  
  
  run_experiment(args)
