# /// script
# dependencies = ["wandb"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Image_Segmentation_with_Keras.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{keras-segmentation} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports and Setups
    <!--- @wandbcode{keras-segmentation} -->
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb !pip install -qq wandb
    return


@app.cell
def _():
    import wandb
    from wandb.keras import WandbMetricsLogger
    from wandb.keras import WandbEvalCallback

    return WandbEvalCallback, WandbMetricsLogger, wandb


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell
def _():
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras import models

    import tensorflow_datasets as tfds

    import os
    import numpy as np
    from argparse import Namespace
    import matplotlib.pyplot as plt

    return Namespace, layers, models, np, tf, tfds


@app.cell
def _(Namespace):
    configs = Namespace(
        img_size = 128,
        batch_size = 32,
        num_classes = 3,
    )
    configs
    return (configs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dataloader

    We will be using Oxford Pets Dataset which we can directly get from TensorFlow Datasets.
    """)
    return


@app.cell
def _(tfds):
    train_ds, valid_ds = tfds.load('oxford_iiit_pet', split=["train", "test"])
    return train_ds, valid_ds


@app.cell
def _(configs, tf, train_ds, valid_ds):
    AUTOTUNE = tf.data.experimental.AUTOTUNE


    def parse_data(example):
        # Parse image
        image = example["image"]
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, size=(configs.img_size, configs.img_size))

        # Parse mask
        mask = example["segmentation_mask"] - 1 # ground truth labels are [1,2,3].
        mask = tf.image.resize(mask, size=(configs.img_size, configs.img_size), method='nearest')
        mask = tf.one_hot(tf.squeeze(mask, axis=-1), depth=configs.num_classes)

        return image, mask

    trainloader = (
        train_ds
        .shuffle(1024)
        .map(parse_data, num_parallel_calls=AUTOTUNE)
        .batch(configs.batch_size)
        .prefetch(AUTOTUNE)
    )

    validloader = (
        valid_ds
        .map(parse_data, num_parallel_calls=AUTOTUNE)
        .batch(configs.batch_size)
        .prefetch(AUTOTUNE)
    )
    return trainloader, validloader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model
    """)
    return


@app.cell
def _(configs, layers, models):
    # ref: https://github.com/ayulockin/deepimageinpainting/blob/master/Image_Inpainting_Autoencoder_Decoder_v2_0.ipynb
    class SegmentationModel:
        '''
        Build UNET based model for segmentation task.
        '''
        def prepare_model(self, OUTPUT_CHANNEL, input_size=(configs.img_size, configs.img_size, 3)):
            inputs = layers.Input(input_size)

            conv1, pool1 = self.__ConvBlock(32, (3,3), (2,2), 'relu', 'same', inputs)
            conv2, pool2 = self.__ConvBlock(64, (3,3), (2,2), 'relu', 'same', pool1)
            conv3, pool3 = self.__ConvBlock(128, (3,3), (2,2), 'relu', 'same', pool2)
            conv4, pool4 = self.__ConvBlock(256, (3,3), (2,2), 'relu', 'same', pool3)
        
            conv5, up6 = self.__UpConvBlock(512, 256, (3,3), (2,2), (2,2), 'relu', 'same', pool4, conv4)
            conv6, up7 = self.__UpConvBlock(256, 128, (3,3), (2,2), (2,2), 'relu', 'same', up6, conv3)
            conv7, up8 = self.__UpConvBlock(128, 64, (3,3), (2,2), (2,2), 'relu', 'same', up7, conv2)
            conv8, up9 = self.__UpConvBlock(64, 32, (3,3), (2,2), (2,2), 'relu', 'same', up8, conv1)

            conv9 = self.__ConvBlock(32, (3,3), (2,2), 'relu', 'same', up9, False)
        
            outputs = layers.Conv2D(OUTPUT_CHANNEL, (3, 3), activation='softmax', padding='same')(conv9)

            return models.Model(inputs=[inputs], outputs=[outputs])  

        def __ConvBlock(self, filters, kernel_size, pool_size, activation, padding, connecting_layer, pool_layer=True):
            conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(connecting_layer)
            conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(conv)
            if pool_layer:
              pool = layers.MaxPooling2D(pool_size)(conv)
              return conv, pool
            else:
              return conv

        def __UpConvBlock(self, filters, up_filters, kernel_size, up_kernel, up_stride, activation, padding, connecting_layer, shared_layer):
            conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(connecting_layer)
            conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(conv)
            up = layers.Conv2DTranspose(filters=up_filters, kernel_size=up_kernel, strides=up_stride, padding=padding)(conv)
            up = layers.concatenate([up, shared_layer], axis=3)

            return conv, up

    return (SegmentationModel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Initialize Model and Compile
    """)
    return


@app.cell
def _(SegmentationModel, configs, tf):
    # output channel is 3 because we have three classes in our mask
    tf.keras.backend.clear_session()
    model = SegmentationModel().prepare_model(configs.num_classes)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
    )

    model.summary()
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Callback
    """)
    return


@app.cell
def _():
    segmentation_classes = ['pet', 'pet_outline', 'background']

    # returns a dictionary of labels
    def labels():
        l = {}
        for i, label in enumerate(segmentation_classes):
            l[i] = label
        return l

    return (labels,)


@app.cell
def _(WandbEvalCallback, labels, np, tf, wandb):
    class WandbSemanticLogger(WandbEvalCallback):
        def __init__(
            self,
            validloader,
            data_table_columns=["index", "image"],
            pred_table_columns=["epoch", "index", "image", "prediction"],
            num_samples=100,
        ):
            super().__init__(
                data_table_columns,
                pred_table_columns,
            )

            self.val_data = validloader.unbatch().take(num_samples)

        def add_ground_truth(self, logs):
            for idx, (image, mask) in enumerate(self.val_data):
                self.data_table.add_data(
                    idx,
                    self._prepare_wandb_mask(
                        image.numpy(),
                        np.argmax(mask.numpy(), axis=-1),
                        "ground_truth"
                    )
                )

        def add_model_predictions(self, epoch, logs):
            data_table_ref = self.data_table_ref
            table_idxs = data_table_ref.get_index()

            for idx, (image, mask) in enumerate(self.val_data):
                prediction = self.model.predict(tf.expand_dims(image, axis=0), verbose=0)
                prediction = np.argmax(tf.squeeze(prediction, axis=0).numpy(), axis=-1)

                self.pred_table.add_data(
                    epoch,
                    data_table_ref.data[idx][0],
                    self._prepare_wandb_mask(
                        data_table_ref.data[idx][1],
                        np.argmax(mask.numpy(), axis=-1),
                        "ground_truth"
                    ),
                    self._prepare_wandb_mask(
                        data_table_ref.data[idx][1],
                        prediction,
                        "prediction"
                    )
                )

        def _prepare_wandb_mask(self, image, mask, mask_type):
            return wandb.Image(
                image,
                masks = {
                    "ground_truth": {
                        "mask_data": mask,
                        "class_labels": labels()
                }})

    return (WandbSemanticLogger,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train
    """)
    return


@app.cell
def _(
    WandbMetricsLogger,
    WandbSemanticLogger,
    configs,
    model,
    trainloader,
    validloader,
    wandb,
):
    run = wandb.init(project='image-segmentation', config=configs)

    _ = model.fit(
        trainloader, 
        epochs=10, 
        validation_data=validloader,
        callbacks=[
            WandbMetricsLogger(log_freq=2),
            WandbSemanticLogger(validloader)
          ]
        )

    run.finish()
    return


if __name__ == "__main__":
    app.run()
