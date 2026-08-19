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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Use_WandbModelCheckpoint_in_your_Keras_workflow.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{intro-colab-keras-metricslogger} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    <!--- @wandbcode{intro-colab-keras-metricslogger} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Using Keras Checkpoint callback with Weights & Biases
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use Weights & Biases for machine learning experiment tracking, dataset versioning, and project collaboration.

    <img src="http://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This colab notebook introduces the `WandbModelCheckpoint` callback. Use this callback to log your model checkpoints to Weight and Biases [Artifacts](https://docs.wandb.ai/guides/data-and-model-versioning).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🌴 Setup and Installation

    First, let us install the latest version of Weights and Biases. We will then authenticate this colab instance to use W&B.
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb !pip install -qq -U wandb
    return


@app.cell
def _():
    import os
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras import models
    import tensorflow_datasets as tfds

    # Weights and Biases related imports
    import wandb
    from wandb.integration.keras import WandbMetricsLogger
    from wandb.integration.keras import WandbModelCheckpoint

    return (
        WandbMetricsLogger,
        WandbModelCheckpoint,
        layers,
        models,
        tf,
        tfds,
        wandb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If this is your first time using W&B or you are not logged in, the link that appears after running `wandb.login()` will take you to sign-up/login page. Signing up for a [free account](https://wandb.ai/signup) is as easy as a few clicks.
    """)
    return


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🌳 Hyperparameters

    Use of proper config system is a recommended best practice for reproducible machine learning. We can track the hyperparameters for every experiment using W&B. In this colab we will be using simple Python `dict` as our config system.
    """)
    return


@app.cell
def _():
    configs = dict(
        num_classes = 10,
        shuffle_buffer = 1024,
        batch_size = 64,
        image_size = 28,
        image_channels = 1,
        earlystopping_patience = 3,
        learning_rate = 1e-3,
        epochs = 10
    )
    return (configs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🍁 Dataset

    In this colab, we will be using [Fashion-MNIST](https://www.tensorflow.org/datasets/catalog/fashion_mnist) dataset from TensorFlow Dataset catalog. We aim to build a simple image classification pipeline using TensorFlow/Keras.
    """)
    return


@app.cell
def _(tfds):
    train_ds, valid_ds = tfds.load('fashion_mnist', split=['train', 'test'])
    return train_ds, valid_ds


@app.cell
def _(configs, tf):
    AUTOTUNE = tf.data.AUTOTUNE


    def parse_data(example):
        # Get image
        image = example["image"]
        # image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        # Get label
        label = example["label"]
        label = tf.one_hot(label, depth=configs["num_classes"])

        return image, label


    def get_dataloader(ds, configs, dataloader_type="train"):
        dataloader = ds.map(parse_data, num_parallel_calls=AUTOTUNE)

        if dataloader_type=="train":
            dataloader = dataloader.shuffle(configs["shuffle_buffer"])
      
        dataloader = (
            dataloader
            .batch(configs["batch_size"])
            .prefetch(AUTOTUNE)
        )

        return dataloader

    return (get_dataloader,)


@app.cell
def _(configs, get_dataloader, train_ds, valid_ds):
    trainloader = get_dataloader(train_ds, configs)
    validloader = get_dataloader(valid_ds, configs, dataloader_type="valid")
    return trainloader, validloader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🎄 Model
    """)
    return


@app.cell
def _(layers, models, tf):
    def get_model(configs):
        backbone = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False)
        backbone.trainable = False

        inputs = layers.Input(shape=(configs["image_size"], configs["image_size"], configs["image_channels"]))
        resize = layers.Resizing(32, 32)(inputs)
        neck = layers.Conv2D(3, (3,3), padding="same")(resize)
        preprocess_input = tf.keras.applications.mobilenet.preprocess_input(neck)
        x = backbone(preprocess_input)
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(configs["num_classes"], activation="softmax")(x)

        return models.Model(inputs=inputs, outputs=outputs)

    return (get_model,)


@app.cell
def _(configs, get_model, tf):
    tf.keras.backend.clear_session()
    model = get_model(configs)
    model.summary()
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🌿 Compile Model
    """)
    return


@app.cell
def _(model, tf):
    model.compile(
        optimizer = "adam",
        loss = "categorical_crossentropy",
        metrics = ["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top@5_accuracy')]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🌻 Train
    """)
    return


@app.cell
def _(
    WandbMetricsLogger,
    WandbModelCheckpoint,
    configs,
    model,
    trainloader,
    validloader,
    wandb,
):
    # Initialize a W&B run
    run = wandb.init(
        project = "intro-keras",
        config = configs
    )

    # Train your model
    model.fit(
        trainloader,
        epochs = configs["epochs"],
        validation_data = validloader,
        callbacks = [
            WandbMetricsLogger(log_freq=10),
            WandbModelCheckpoint(filepath="models/model.keras") # Notice the use of WandbModelCheckpoint here
        ]
    )

    # Close the W&B run
    run.finish()
    return


if __name__ == "__main__":
    app.run()
