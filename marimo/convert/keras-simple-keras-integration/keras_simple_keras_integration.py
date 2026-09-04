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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Simple_Keras_Integration.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{keras-video} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    <!--- @wandbcode{keras-video} -->

    # W&B 💘 Keras
    Use Weights & Biases for machine learning experiment tracking, model checkpointing, and project collaboration.

    <img src="https://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

    ## Keras Documentation

    For a full guide of how to log to Weights & Biases using Keras, see **[our Keras documentation](https://docs.wandb.ai/guides/integrations/keras)**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What this notebook covers:

    We show you how to integrate Weights & Biases with your Keras code to add experiment tracking to your pipeline. That includes:

    1. Storing hyperparameters and metadata in a `config`.
    2. Passing the wandb Keras callbacks to `model.fit`. This will automatically log training metrics, like loss, and system metrics, like GPU and CPU utilization.
    3. Using the `wandb.log` API to log custom metrics.

    all using the CIFAR-10 dataset.

    Then, we'll show you how to catch your model making mistakes by logging both the output predictions and the input images the network used to generate them.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Follow along with a [video tutorial](http://wandb.me/keras-video)!
    **Note**: Sections starting with _Step_ are all you need to integrate W&B in an existing pipeline. The rest just loads data and defines a model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Install, Import, and Log In
    """)
    return


@app.cell
def _():
    import os
    import random

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras import models
    from tensorflow.keras.datasets import cifar10
    import tensorflow_datasets as tfds

    return cifar10, keras, layers, models, np, tf, tfds


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 0: Install W&B
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb !pip install -qU wandb
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 1: Import W&B and Login
    """)
    return


@app.cell
def _():
    import wandb
    from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint, WandbEvalCallback

    return WandbEvalCallback, WandbMetricsLogger, WandbModelCheckpoint, wandb


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > Side note: If this is your first time using W&B or you are not logged in, the link that appears after running `wandb.login` will take you to sign-up/login page. Signing up is easy!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Download and Prepare the Dataset
    """)
    return


@app.cell
def _(cifar10):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Subsetting train data and normalizing to [0., 1.]
    x_train, x_test = x_train[::5] / 255., x_test / 255.
    y_train = y_train[::5]

    CLASS_NAMES = ["airplane", "automobile", "bird", "cat",
                   "deer", "dog", "frog", "horse", "ship", "truck"]

    print('Shape of x_train: ', x_train.shape)
    print('Shape of y_train: ', y_train.shape)
    print('Shape of x_test: ', x_test.shape)
    print('Shape of y_test: ', y_test.shape)
    return CLASS_NAMES, x_test, x_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Define the Model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here, we define a standard CNN (with convolution and max-pooling) in Keras.
    """)
    return


@app.cell
def _(CLASS_NAMES, keras):
    def Model():
      inputs = keras.layers.Input(shape=(32, 32, 3))

      x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(inputs)
      x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
      x = keras.layers.MaxPooling2D(pool_size=2)(x)

      x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
      x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)

      x = keras.layers.GlobalAveragePooling2D()(x)

      x = keras.layers.Dense(128, activation='relu')(x)
      x = keras.layers.Dense(32, activation='relu')(x)
  
      outputs = keras.layers.Dense(len(CLASS_NAMES), activation='softmax')(x)

      return keras.models.Model(inputs=inputs, outputs=outputs)

    return (Model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Train the Model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 2: Give `wandb.init` your `config`

    You first initialize your wandb run, letting us know some training is about to happen. [Check out the official documentation for `.init` here $\rightarrow$](https://docs.wandb.com/library/init)

    That's when you need to set your hyperparameters.
    They're passed in as a dictionary via the `config` argument,
    and then become available as the `config` attribute of `wandb`.

    Learn more about `config` in this [Colab Notebook $\rightarrow$](http://wandb.me/config-colab)
    """)
    return


@app.cell
def _(Model, tf, wandb):
    # Initialize wandb with your project name
    run = wandb.init(project='my-keras-project',
                     config={  # and include hyperparameters and metadata
                         "learning_rate": 0.005,
                         "epochs": 5,
                         "batch_size": 1024,
                         "loss_function": "sparse_categorical_crossentropy",
                         "architecture": "CNN",
                         "dataset": "CIFAR-10"
                     })
    config = wandb.config  # We'll use this to configure our experiment

    # Initialize model like you usually do.
    tf.keras.backend.clear_session()
    model = Model()
    model.summary()

    # Compile model like you usually do.
    # Notice that we use config, so our metadata matches what gets executed
    optimizer = tf.keras.optimizers.Adam(config.learning_rate) 
    model.compile(optimizer, config.loss_function, metrics=['acc'])
    return config, model, run


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 3: Pass `WandbMetricsLogger` and `WandbModelCheckpoint` to `model.fit`

    Keras has a [robust callbacks system](https://keras.io/api/callbacks/) that
    allows users to separate model definition and the core training logic
    from other behaviors that occur during training and testing.

    That includes, for example,

    **Click on the Project page link above to see your results!**
    """)
    return


@app.cell
def _(
    WandbMetricsLogger,
    WandbModelCheckpoint,
    config,
    model,
    x_test,
    x_train,
    y_test,
    y_train,
):
    # Add WandbMetricsLogger to log metrics and WandbModelCheckpoint to log model checkpoints
    _wandb_callbacks = [WandbMetricsLogger(), WandbModelCheckpoint(filepath='my_model_{epoch:02d}')]
    model.fit(x_train, y_train, epochs=config.epochs, batch_size=config.batch_size, validation_data=(x_test, y_test), callbacks=_wandb_callbacks)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Use `wandb.log` for custom metrics

    Here, we log the error rate on the test set.
    """)
    return


@app.cell
def _(model, run, wandb, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test)
    print('Test Error Rate: ', round((1 - accuracy) * 100, 2))

    # With wandb.log, we can easily pass in metrics as key-value pairs.
    wandb.log({'Test Error Rate': round((1 - accuracy) * 100, 2)})

    run.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Log predictions on test data using `WandbEvalCallback`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `WandbEvalCallback` is an abstract base class to build Keras callbacks for primarily model prediction visualization and secondarily dataset visualization.

    This is a dataset and task agnostic abstract callback. To use this, inherit from this base callback class and implement the `add_ground_truth` and `add_model_prediction` methods.

    The `WandbEvalCallback` is a utility class that provides helpful methods to:

    - create data and prediction `wandb.Table` instances,
    - log data and prediction Tables as `wandb.Artifact`,
    - logs the data table `on_train_begin`,
    - logs the prediction table `on_epoch_end`.

    As an example, we have implemented `WandbClsEvalCallback` below for an image classification task. This example callback:
    - logs the validation data (`data_table`) to W&B,
    - performs inference and logs the prediction (`pred_table`) to W&B on every epoch end.

    ## How the memory footprint is reduced?

    We log the `data_table` to W&B when the `on_train_begin` method is ivoked. Once it's uploaded as a W&B Artifact, we get a reference to this table which can be accessed using `data_table_ref` class variable. The `data_table_ref` is a 2D list that can be indexed like `self.data_table_ref[idx][n]` where `idx` is the row number while `n` is the column number. Let's see the usage in the example below.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Sub-class `WandbEvalCallback`
    """)
    return


@app.cell
def _(WandbEvalCallback, logits, np, tf, wandb):
    class WandbClsEvalCallback(WandbEvalCallback):
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
                logit = logits[idx]
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

    return (WandbClsEvalCallback,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Create Dataset processing and Dataloaders functions
    """)
    return


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Define our model
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Set our config
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
    ### Dataset

    In this example, we will be using [CIFAR100](https://www.tensorflow.org/datasets/catalog/cifar100) dataset from TensorFlow Dataset catalog. We aim to build a simple image classification pipeline using TensorFlow/Keras.
    """)
    return


@app.cell
def _(tfds):
    train_ds, valid_ds = tfds.load('fashion_mnist', split=['train', 'test'])
    return train_ds, valid_ds


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Create our Dataloaders and Model
    """)
    return


@app.cell
def _(configs, get_dataloader, train_ds, valid_ds):
    trainloader = get_dataloader(train_ds, configs)
    validloader = get_dataloader(valid_ds, configs, dataloader_type="valid")
    return trainloader, validloader


@app.cell
def _(configs, get_model, tf):
    tf.keras.backend.clear_session()
    model_1 = get_model(configs)
    model_1.summary()
    return (model_1,)


@app.cell
def _(model_1, tf):
    model_1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top@5_accuracy')])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Train the model and log the predictions to a W&B Table
    """)
    return


@app.cell
def _(
    WandbClsEvalCallback,
    WandbMetricsLogger,
    WandbModelCheckpoint,
    configs,
    model_1,
    trainloader,
    validloader,
    wandb,
):
    run_1 = wandb.init(project='my-keras-project', config=configs)
    _wandb_callbacks = [WandbMetricsLogger(log_freq=10), WandbModelCheckpoint(filepath='my_model_{epoch:02d}'), WandbClsEvalCallback(validloader, data_table_columns=['idx', 'image', 'ground_truth'], pred_table_columns=['epoch', 'idx', 'image', 'ground_truth', 'prediction'])]
    model_1.fit(trainloader, epochs=configs['epochs'], validation_data=validloader, callbacks=_wandb_callbacks)
    run_1.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Click on the **W&B project page** link above to see your live results.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Whats Next? Hyperparameters with Sweeps

    We tried out two different hyperparameter settings by hand. You can use Weights & Biases Sweeps to automate hyperparameter testing and explore the space of possible models and optimization strategies.

    ## [Check out Hyperparameter Optimization in TensorFlow uisng W&B Sweep $\rightarrow$](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/tensorflow/Hyperparameter_Optimization_in_TensorFlow_using_W&B_Sweeps.ipynb)

    Running a hyperparameter sweep with Weights & Biases is very easy. There are just 3 simple steps:

    1. **Define the sweep:** We do this by creating a dictionary or a [YAML file](https://docs.wandb.com/library/sweeps/configuration) that specifies the parameters to search through, the search strategy, the optimization metric et all.

    2. **Initialize the sweep:**
    `sweep_id = wandb.sweep(sweep_config)`

    3. **Run the sweep agent:**
    `wandb.agent(sweep_id, function=train)`

    And voila! That's all there is to running a hyperparameter sweep! In the notebook below, we'll walk through these 3 steps in more detail.

    <img src="https://imgur.com/UiQKg0L.png" alt="Weights & Biases" />
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Documentation

    For a full guide of how to log to Weights & Biases using Keras, see **[our Keras documentation](https://docs.wandb.ai/guides/integrations/keras)**
    """)
    return


if __name__ == "__main__":
    app.run()
