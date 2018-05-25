#!/usr/bin/env python

import numpy as np
import scipy.misc

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, \
    Conv2DTranspose, Reshape, AveragePooling2D, UpSampling2D
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, adam
from keras.callbacks import LambdaCallback
from keras.layers.advanced_activations import LeakyReLU
from keras import initializers
from keras import metrics
import keras
from os import path
import scipy.misc

import wandb
from wandb.keras import WandbCallback
from keras.models import load_model
import argparse

wandb.init()
config = wandb.config

config.adversarial_epochs = 1000
config.discriminator_epochs = 1
config.discriminator_examples = 10000
config.generator_epochs = 12
config.generator_examples = 10000
config.generator_seed_dim = 10
config.generator_conv_size = 64
config.batch_size = 100
config.image_shape = (28, 28, 1)

def add_noise(labels):
    for label in labels:
        noise = np.random.uniform(0.0,0.3)
        if label[0] == 0.0:
            label[0]+= noise
            label[1]-=noise
        else:
            label[0]-=noise
            label[1]+=noise
        if np.random.uniform(0,1) > 0.05:
            tmp = label[0]
            label[0] = label[1]
            label[1] = tmp

def mix_data(data, generator, length=1000):
    num_examples=int(length/2)

    data= data[:num_examples, :, :]


    seeds = np.random.normal(0, 1, (num_examples, config.generator_seed_dim))

    fake_train = generator.predict(seeds)[:,:,:,0]

    combined  = np.concatenate([ data, fake_train ])

    # combine them together
    labels = np.zeros(combined.shape[0])
    labels[:data.shape[0]] = 1

    indices = np.arange(combined.shape[0])
    np.random.shuffle(indices)
    combined = combined[indices]
    labels = labels[indices]
    combined.shape += (1,)

    labels = np_utils.to_categorical(labels)

    add_noise(labels)

    return (combined, labels)

def log_discriminator(epoch, logs):
    wandb.log({
            'generator_loss': 0.0,
            'generator_acc': (1.0-logs['acc'])*2.0,
            'discriminator_loss': logs['loss'],
            'discriminator_acc': logs['acc']})

def create_discriminator():

    discriminator = Sequential()
    discriminator.add(Flatten(input_shape=config.image_shape))
    discriminator.add(Dense(1024, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(2, activation='sigmoid'))
    discriminator.compile(optimizer='sgd', loss='categorical_crossentropy',
        metrics=['acc'])
    return discriminator

def create_generator():
    random_dim = config.generator_seed_dim

    generator = Sequential()
    generator.add(Dense(256, input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(784, activation='tanh'))
    generator.add(Reshape(config.image_shape))
    generator.compile(loss='categorical_crossentropy', optimizer='adam')

    return generator

def create_joint_model(generator, discriminator):
    joint_model = Sequential()
    joint_model.add(generator)
    joint_model.add(discriminator)

    discriminator.trainable = False

    joint_model.compile(optimizer='adam', loss='categorical_crossentropy',
        metrics=['acc'])

    return joint_model


def generator_inputs(num_examples):
    return np.random.normal(0, 1, (num_examples, config.generator_seed_dim))


def train_discriminator(generator, discriminator, x_train, x_test, iter):

    train, train_labels = mix_data(x_train, generator, config.discriminator_examples)
    test, test_labels = mix_data(x_test, generator, config.discriminator_examples)

    discriminator.trainable = True
    discriminator.summary()

    wandb_logging_callback = LambdaCallback(on_epoch_end=log_discriminator)

    history = discriminator.fit(train, train_labels,
        epochs=config.discriminator_epochs,
        batch_size=config.batch_size, validation_data=(test, test_labels),
        callbacks = [wandb_logging_callback])

    discriminator.save(path.join(wandb.run.dir, "discriminator.h5"))

def log_generator(epoch, logs):
    wandb.log({'generator_loss': logs['loss'],
                     'generator_acc': logs['acc'],
                     'discriminator_loss': 0.0,
                     'discriminator_acc': (1-logs['acc'])/2.0+0.5})


def train_generator(generator, discriminator, joint_model):
    num_examples = config.generator_examples

    train = generator_inputs(num_examples)
    labels = np_utils.to_categorical(np.ones(num_examples))

    add_noise(labels)

    wandb_logging_callback = LambdaCallback(on_epoch_end=log_generator)

    discriminator.trainable = False

    joint_model.summary()

    joint_model.fit(train, labels, epochs=config.generator_epochs,
            batch_size=config.batch_size,
            callbacks=[wandb_logging_callback])

    generator.save(path.join(wandb.run.dir, "generator.h5"))


def sample_images(generator):
    noise = generator_inputs(10)
    gen_imgs = generator.predict(noise)
    wandb.log({'examples': [wandb.Image(np.squeeze(i)) for i in gen_imgs]})


def main():
    parser = argparse.ArgumentParser(description='Wandb example GAN')
    parser.add_argument('--disc', type=str, default=None, metavar='N',
                        help='link to discriminator model file')
    parser.add_argument('--gen', type=str, default=None, metavar='N',
                        help='link to generator model file')
    args = parser.parse_args()

    # load the real training data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0 * 2.0 - 1.0
    x_test = x_test / 255.0 * 2.0 - 1.0
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    # only try to create 8s
    # x_train = x_train[y_train == 8]
    # x_test = x_test[y_test == 8]
    if args.disc:
        discriminator = load_model(args.disc)
    else:
        discriminator = create_discriminator()

    if args.gen:
        generator = load_model(args.gen)
    else:
        generator = create_generator()

    joint_model = create_joint_model(generator, discriminator)

    for i in range(config.adversarial_epochs):
        train_discriminator(generator, discriminator, x_train, x_test, i)
        train_generator(generator, discriminator, joint_model)
        sample_images(generator)

if __name__ == '__main__':
   main()
