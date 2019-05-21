#!/usr/bin/env python3

from model import unet
import wandb
import os
from wandb.keras import WandbCallback
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
import keras_metrics as km

from data import get_training_data

augmentation_params = {
    'rotation_range': 0.2,
    'width_shift_range': 0.05,
    'height_shift_range': 0.05,
    'shear_range': 0.05,
    'zoom_range': 0.05,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

def main():
    run = wandb.init(
        project='witness-puzzle-finder-3',
        tensorboard=True)

    wandb.save('*.py')

    run.config.learning_rate = 1e-4
    run.config.num_epochs = 50
    run.config.steps_per_epoch = 300
    run.config.batch_size = 8
    run.config.image_size = (288, 512)
    run.config.num_predictions = 24

    training_data_generator = get_training_data(run.config.batch_size,
        'data/train', 'images', 'labels', augmentation_params,
        target_size=run.config.image_size)

    validation_data_generator = get_training_data(run.config.batch_size,
        'data/valid', 'images', 'labels', {}, target_size=run.config.image_size)

    validation_data_generator_2 = get_training_data(run.config.batch_size,
        'data/valid', 'images', 'labels', {}, target_size=run.config.image_size)

    try:
        os.makedirs('model')
    except OSError:
        pass

    model = unet(image_size=run.config.image_size)
    metrics = ['accuracy', km.precision(), km.recall()]

    model.compile(
        optimizer=Adam(lr=run.config.learning_rate),
        loss='binary_crossentropy',
        metrics=metrics)

    model_checkpoint = ModelCheckpoint(
        'model/unet_witness.hdf5', monitor='loss',
        verbose=1, save_best_only=True)

    wandb_callback = WandbCallback(
        data_type='image',
        predictions=run.config.num_predictions,
        generator=validation_data_generator_2,
        save_model=False,
        labels=['void', 'puzzle'])

    tensorboard_callback = TensorBoard(
        log_dir=wandb.run.dir,
        histogram_freq=0,
        write_graph=True,
        write_images=True)

    callbacks = [model_checkpoint, wandb_callback, tensorboard_callback]

    model.fit_generator(
        training_data_generator,
        validation_data=validation_data_generator,
        validation_steps=run.config.num_predictions,
        steps_per_epoch=run.config.steps_per_epoch,
        epochs=run.config.num_epochs,
        callbacks=callbacks)

if __name__ == '__main__':
    main()
