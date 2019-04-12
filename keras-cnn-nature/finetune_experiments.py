import argparse
import os

# keras pre-trained models
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2 as IRV2
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception

from keras.layers import Dense, GlobalAveragePooling2D, Conv2D
from keras.layers import MaxPooling2D, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.models import Model, Sequential
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

import wandb
from wandb.keras import WandbCallback

# training config
#--------------------------------
img_width = 299
img_height = 299
resnet_img_dim = 224

# Model definitions
#--------------------------------
BASE_MODELS = {
  "irv2" : IRV2,
  "iv3" : InceptionV3,
  "resnet" : ResNet50,
  "xception" : Xception
}

def build_small_cnn(dropout, num_classes, lr):
  """ Build a small 7-layer convnet """
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

  lr_optimizer = optimizers.SGD(lr=lr, momentum=0.8)
  model.compile(loss='categorical_crossentropy',
                optimizer=lr_optimizer,
                metrics=['accuracy'])
  return model

def load_pretrained_model(base_model_name, fc_size, num_classes):
  """ Load pre-trained base network, add a global average pooling layer and two
  fully connected layers for fine-tuning, and freeze all base layers """
  base_model = BASE_MODELS[base_model_name]
  base = base_model(weights='imagenet', include_top=False)
  x = base.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(fc_size, activation='relu')(x)
  guesses = Dense(num_classes, activation='softmax')(x)
  model = Model(inputs=base.input, outputs=guesses)
  
  # freeze all base layers
  for layer in base.layers:
    layer.trainable = False
  
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
  return model

def log_model_params(model, wandb_config, args, img_dim):
  """ Extract params of interest about the model (e.g. number of different layer types).
  Log these and any experiment-level settings to wandb """
  wandb_config.update({
    "epochs" : args.epochs,
    "batch_size" : args.batch_size,
    "img_dim" : img_dim,
    "num_classes" : args.num_classes,
    "n_train" : args.num_train,
    "n_valid" : args.num_valid,
    "fc_size" : args.fc_size,
    "pre_epochs" : args.pretrain_epochs,
    "lr" : args.learning_rate,
    "mnt" : args.momentum,
    "freeze_layer" : args.freeze_layer,
    "base_model" : args.initial_model 
  })

# Finetuning experiments with different base models, 
# number of pretraining epochs, and number of layers to freeze before finetuning

def finetune_base_cnn(args):
  """ Load a pre-trained model and pre-train it for some epochs (args.pretrain_epochs).
  Then freeze learned layers up to args.freeze_layer, and continue training the remaining
  layers for the rest of the epochs (args.epochs) """
  wandb.init(project=args.project_name)
  callbacks = [WandbCallback()]
  
  # basic data augmentation
  train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
  test_datagen = ImageDataGenerator(rescale=1. / 255)

  # modify image dims depending on base model (only resnet is different)
  if args.initial_model == "resnet":
    base_img_width = resnet_img_dim
    base_img_height = resnet_img_dim
  else:
    base_img_width = img_width
    base_img_height = img_height

  train_generator = train_datagen.flow_from_directory(
    args.train_data,
    target_size=(base_img_width, base_img_height),
    batch_size=args.batch_size,
    class_mode='categorical',
    follow_links=True)

  validation_generator = test_datagen.flow_from_directory(
    args.val_data,
    target_size=(base_img_width, base_img_height),
    batch_size=args.batch_size,
    class_mode='categorical',
    follow_links=True)

  model = load_pretrained_model(args.initial_model, args.fc_size, args.num_classes)
  log_model_params(model, wandb.config, args, base_img_width)
   
  # Pre-training phase 
  #-----------------------
  model.fit_generator(
    train_generator,
    steps_per_epoch=args.num_train // args.batch_size,
    epochs=args.pretrain_epochs,
    validation_data=validation_generator,
    callbacks = callbacks,
    validation_steps=args.num_valid // args.batch_size)

  # optionally show all layers of the base model
  #for i, layer in enumerate(model.layers):
  #  print i, layer.name
  
  # freeze layers up to the freeze_layer index
  for layer in model.layers[:args.freeze_layer]:
    layer.trainable = False
  for layer in model.layers[args.freeze_layer:]:
    layer.trainable = True

  # recompile model
  model.compile(optimizer=optimizers.SGD(lr=args.learning_rate, momentum=args.momentum), loss='categorical_crossentropy', metrics=["accuracy"])

  # Finetuning phase
  #-----------------------  
  model.fit_generator(
    train_generator,
    steps_per_epoch=args.num_train // args.batch_size,
    epochs=args.epochs,
    validation_data=validation_generator,
    callbacks = callbacks,
    validation_steps=args.num_valid // args.batch_size)

  save_model_filename = args.model_name + ".h5"
  model.save_weights(save_model_filename)

# Curriculum learning experiments with number of pretraining (on more general/class labels)
# versus finetuning (on more specific/species labels) epochs and learning rates for general
# and specific phases

def curr_learn_experiment(args):
  """ Run curriculum learning experiment, pre-training on class labels for
  args.class_switch total epochs, then finetuning on species labels for 
  args.epochs total epochs """
  wandb.init(project=args.project_name)

  # NOTE: these absolute paths to the general and specific train and validation
  # data depend on your setup 
  general_train = "curr_learn_25_s_620_100_BY_CLASS/train"
  general_val = "curr_learn_25_s_620_100_BY_CLASS/val"
  specific_train = "curr_learn_25_s_620_100/train"
  specific_val = "curr_learn_25_s_620_100/val"
  
  train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
  test_datagen = ImageDataGenerator(rescale=1. / 255) 
  
  # initially, the more general model pre-trains on 5 classes
  # (amphibians, birds, insects, mammals, and reptiles)
  # on data generated with general labels (biological/taxonomic class)
  general_model = build_small_cnn(args.dropout, 5, args.class_lr)
  log_model_params(general_model, wandb.config, args, img_width)

  switch_epochs = args.class_switch
  callbacks = [WandbCallback()]
  
  train_generator = train_datagen.flow_from_directory(
    general_train,
    target_size=(img_width, img_height),
    batch_size=args.batch_size,
    class_mode='categorical',
    follow_links=True)

  validation_generator = test_datagen.flow_from_directory(
    general_val,
    target_size=(img_width, img_height),
    batch_size=args.batch_size,
    class_mode='categorical',
    follow_links=True)
  
  # pre-train on first, general label set (5 classes) for switch_epochs
  general_model.fit_generator(
    train_generator,
    steps_per_epoch=args.num_train // args.batch_size,
    epochs=switch_epochs,
    validation_data=validation_generator,
    callbacks=callbacks,
    validation_steps=args.num_valid // args.batch_size)

  # recompile the model such that the final layer can predict 25 output labels (25 species)
  specific_model = build_small_cnn(args.dropout, 25, args.species_lr)
  # copy weights from general to specific model, up until dropout layer
  for i, layer in enumerate(specific_model.layers[:-3]):
    layer.set_weights(general_model.layers[i].get_weights())

  # finetune on second, specific label set (25 species) for finetune_epochs,
  # on data generated with specific labels (biological/taxonomic species)
  finetune_epochs = args.epochs - switch_epochs
  spec_train_generator = train_datagen.flow_from_directory(
    specific_train,
    target_size=(img_width, img_height),
    batch_size=args.batch_size,
    class_mode='categorical',
    follow_links=True)

  spec_validation_generator = test_datagen.flow_from_directory(
    specific_val,
    target_size=(img_width, img_height),
    batch_size=args.batch_size,
    class_mode='categorical',
    follow_links=True)

  specific_model.fit_generator(
    spec_train_generator,
    steps_per_epoch=args.num_train // args.batch_size,
    epochs=finetune_epochs,
    validation_data=spec_validation_generator,
    callbacks=callbacks,
    validation_steps=args.num_valid // args.batch_size)

  save_model_filename = args.model_name + ".h5"
  specific_model.save_weights(save_model_filename)
 
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
  parser.add_argument(
    "-p",
    "--project_name",
    type=str,
    default="finetune_keras",
    help="Name of the W&B project to which results will be logged")

  # Curriculum learning args
  #----------------------------
  parser.add_argument(
    "-cs",
    "--class_switch",
    type=int,
    default=0,
    help="Curr Learn: epoch on which to switch tasks from class to species")
  parser.add_argument(
    "--dropout",
    type=float,
    default=0.3,
    help="Curr Learn: dropout before the last fc layer (0.3)")
  parser.add_argument(
    "--class_lr",
    type=float,
    default=0.025,
    help="Curr Learn: learning rate for class pre-training (0.025)")
  parser.add_argument(
    "--species_lr",
    type=float,
    default=0.01,
    help="Curr Learn: learning rate for species fine-tuning (0.01)")

  # Finetuning args
  #----------------------------
  parser.add_argument(
    "-fc",
    "--fc_size",
    type=int,
    default=1024,
    help="Finetune: size of penultimate fc layer to add onto base net (1024)")
  parser.add_argument(
    "-fl",
    "--freeze_layer",
    type=int,
    default=155,
    help="Finetune: layer of base net up to which we freeze weights (155)")
  parser.add_argument(
    "-i",
    "--initial_model",
    type=str,
    default="iv3",
    help="Finetune: short name of base model to load (iv3, other options: irv2, resnet, xception)")
  parser.add_argument(
    "-pe",
    "--pretrain_epochs",
    type=int,
    default=5,
    help="Finetune: number of pre-training epochs (5)")
  parser.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    default=0.0001,
    help="Finetune: SGD learning rate for finetuning base cnn (0.0001)")
  parser.add_argument(
    "-mnt",
    "--momentum",
    type=float,
    default=0.9,
    help="Finetune: SGD momentum for finetuning base cnn (0.9)")
    
  # Optional args
  #----------------------------
  parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    default=32,
    help="Batch size (32)")
  parser.add_argument(
    "-c",
    "--num_classes",
    type=int,
    default=10,
    help="Number of classes to predict (10)")
  parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    default=45,
    help="Number of training epochs (45)")
  parser.add_argument(
    "-nt",
    "--num_train",
    type=int,
    default=5000,
    help="Number of training examples (5000)")
  parser.add_argument(
    "-nv",
    "--num_valid",
    type=int,
    default=800,
    help="Number of validation examples (800)") 
  parser.add_argument(
    "-d",
    "--train_data",
    type=str,
    default="/mnt/data/inaturalist/main_5000_800/train",
    help="Absolute path to training data")
  parser.add_argument(
    "-v",
    "--val_data",
    type=str,
    default="/mnt/data/inaturalist/main_5000_800/val",
    help="Absolute path to validation data") 

  # wandb tracking utils
  #----------------------------
  parser.add_argument(
    "-q",
    "--dry_run",
    action="store_true",
    help="Dry run (do not log to wandb)")
  parser.add_argument(
    "-t",
    "--tags",
    type=str,
    default="",
    help="tags associated with this run")
  parser.add_argument(
    "-g",
    "--gpu", 
    type=str,
    default="",
    help="if gpu id is set, pass it to the environment vars")
 
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
  
  if args.tags:
    os.environ['WANDB_TAGS'] = args.tags
  if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
  
  # if class_switch is set, run the curriculum learning experiment
  if args.class_switch:
    curr_learn_experiment(args)
  else:
    finetune_base_cnn(args)
