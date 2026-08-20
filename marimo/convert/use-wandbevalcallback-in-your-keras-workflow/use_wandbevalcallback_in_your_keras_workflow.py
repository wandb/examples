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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Use_WandbEvalCallback_in_your_Keras_workflow.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
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
    # Using Keras Evaluation Callbacks with Weights & Biases
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
    This colab notebook introduces the `WandbEvalCallback` which is an abstract callback that be inherited to build useful callbacks for model prediction visualization and dataset visualization. Refer to the [💫 `WandbEvalCallback`](https://colab.research.google.com/drive/107uB39vBulCflqmOWolu38noWLxAT6Be#scrollTo=u50GwKJ70WeJ&line=1&uniqifier=1) section for more details.
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
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras import models
    import tensorflow_datasets as tfds

    # Weights and Biases related imports
    import wandb
    from wandb.integration.keras import WandbMetricsLogger
    from wandb.integration.keras import WandbModelCheckpoint
    from wandb.integration.keras import WandbEvalCallback

    return (
        WandbEvalCallback,
        WandbMetricsLogger,
        layers,
        models,
        np,
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
    # 💫 `WandbEvalCallback`

    The `WandbEvalCallback` is an abstract base class to build Keras callbacks for primarily model prediction visualization and secondarily dataset visualization.

    This is a dataset and task agnostic abstract callback. To use this, inherit from this base callback class and implement the `add_ground_truth` and `add_model_prediction` methods.

    The `WandbEvalCallback` is a utility class that provides helpful methods to:

    - create data and prediction `wandb.Table` instances,
    - log data and prediction Tables as `wandb.Artifact`,
    - logs the data table `on_train_begin`,
    - logs the prediction table `on_epoch_end`.

    As an example, we have implemented `WandbClfEvalCallback` below for an image classification task. This example callback:
    - logs the validation data (`data_table`) to W&B,
    - performs inference and logs the prediction (`pred_table`) to W&B on every epoch end.

    ## How the memory footprint is reduced?

    We log the `data_table` to W&B when the `on_train_begin` method is ivoked. Once it's uploaded as a W&B Artifact, we get a reference to this table which can be accessed using `data_table_ref` class variable. The `data_table_ref` is a 2D list that can be indexed like `self.data_table_ref[idx][n]` where `idx` is the row number while `n` is the column number. Let's see the usage in the example below.
    """)
    return


@app.cell
def _(WandbEvalCallback, np, tf, wandb):
    class WandbClfEvalCallback(WandbEvalCallback):
        def __init__(
            self, validloader, data_table_columns, pred_table_columns, num_samples=100
        ):
            super().__init__(data_table_columns, pred_table_columns)

            self.val_data = validloader.unbatch().take(num_samples)

        def add_ground_truth(self, logs=None):
            for idx, (image, label) in enumerate(self.val_data):
                self.data_table.add_data(
                    idx,
                    wandb.Image(image),
                    np.argmax(label, axis=-1)
                )

        def add_model_predictions(self, epoch, logs=None):
            # Get predictions
            preds = self._inference()
            table_idxs = self.data_table_ref.get_index()

            for idx in table_idxs:
                pred = preds[idx]
                self.pred_table.add_data(
                    epoch,
                    self.data_table_ref.data[idx][0],
                    self.data_table_ref.data[idx][1],
                    self.data_table_ref.data[idx][2],
                    pred
                )

        def _inference(self):
          preds = []
          for image, label in self.val_data:
              pred = self.model(tf.expand_dims(image, axis=0))
              argmax_pred = tf.argmax(pred, axis=-1).numpy()[0]
              preds.append(argmax_pred)

          return preds

    return (WandbClfEvalCallback,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🌻 Train
    """)
    return


@app.cell
def _(
    WandbClfEvalCallback,
    WandbMetricsLogger,
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
            WandbClfEvalCallback(
                validloader,
                data_table_columns=["idx", "image", "ground_truth"],
                pred_table_columns=["epoch", "idx", "image", "ground_truth", "prediction"]
            ) # Notice the use of WandbEvalCallback here
        ]
    )

    # Close the W&B run
    run.finish()
    return


if __name__ == "__main__":
    app.run()
