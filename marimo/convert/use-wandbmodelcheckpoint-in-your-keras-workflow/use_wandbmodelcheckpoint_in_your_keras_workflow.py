# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "importlib-resources==7.1.0",
#     "marimo>=0.23.11",
#     "tensorflow==2.21.0",
#     "tensorflow-datasets==4.9.10",
#     "wandb==0.30.0",
# ]
#
# [tool.marimo.runtime]
# on_cell_change = "lazy"
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Keras checkpoints with Weights & Biases")

with app.setup:
    import marimo as mo
    import os
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from tensorflow.keras import layers, models

    # Weights and Biases related imports
    import wandb
    from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Using WandbModelCheckpoint in your Keras workflow

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/use-wandbmodelcheckpoint-in-your-keras-workflow/use_wandbmodelcheckpoint_in_your_keras_workflow.py/server)

    This notebook introduces the `WandbModelCheckpoint` callback. It saves Keras
    model checkpoints and logs them as versioned Weights & Biases [Artifacts](https://docs.wandb.ai/models/artifacts).
    Use `WandbMetricsLogger` alongside it to track training and validation metrics.

    Use Weights & Biases for machine learning experiment tracking, dataset
    versioning, and project collaboration.

    This notebook is designed to run on your local machine's CPU. The data
    and model cells download Fashion-MNIST and ImageNet weights the first time.

    ## Set up your experiment

    Get your [Forge API key](https://wandb.ai/authorize) and set `WANDB_API_KEY` in
    your runtime's environment or **Secrets** panel. Locally, you can also run
    `wandb login` in a terminal. A hosted runtime needs its own credentials.

    Set `entity` to your Weights & Biases team slug, or leave it as `None` to use your default
    entity. The slug appears in `wandb.ai/<entity>/<project>`. Choose a project
    to group your experiments.
    """)
    return


@app.cell
def _():
    entity = None
    project = "intro-keras"
    wandb.login()
    return entity, project


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Hyperparameters

    Use a Python dictionary to define the experiment. Passing it to
    `wandb.init(config=configs)` records these settings with the run, so you
    can compare experiments and reproduce the model's configuration.
    """)
    return


@app.cell
def _():
    configs = dict(
        dataset="fashion_mnist",
        num_classes=10,
        shuffle_buffer=1024,
        batch_size=64,
        image_size=28,
        image_channels=1,
        learning_rate=0.001,
        epochs=10,
        seed=42,
    )
    return (configs,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Dataset

    We use [Fashion-MNIST](https://www.tensorflow.org/datasets/catalog/fashion_mnist)
    from the TensorFlow Datasets catalog to build a simple image classification
    pipeline. Labels become one-hot vectors; images keep their 0–255 range
    until the model applies MobileNetV2 preprocessing. Validation examples stay
    in a fixed order.

    The data-loading helpers are defined at the end of the notebook.
    """)
    return


@app.cell
def _(configs):
    train_ds, valid_ds = tfds.load(configs["dataset"], split=["train", "test"])
    trainloader = get_dataloader(train_ds, configs)
    validloader = get_dataloader(valid_ds, configs, dataloader_type="valid")
    return trainloader, validloader


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Model

    A frozen ImageNet MobileNetV2 supplies image features. The model resizes each
    grayscale image to 32 × 32, learns a three-channel convolution, applies
    MobileNetV2 preprocessing, and trains a ten-class classification head.
    """)
    return


@app.function
def get_model(configs, weights="imagenet"):
    backbone = tf.keras.applications.mobilenet_v2.MobileNetV2(
        weights=weights, include_top=False
    )
    backbone.trainable = False

    inputs = layers.Input(
        shape=(configs["image_size"], configs["image_size"], configs["image_channels"])
    )
    resize = layers.Resizing(32, 32)(inputs)
    neck = layers.Conv2D(3, (3, 3), padding="same")(resize)
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input(neck)
    x = backbone(preprocess_input, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(configs["num_classes"], activation="softmax")(x)
    return models.Model(inputs=inputs, outputs=outputs)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Compile model

    Adam uses the learning rate recorded in `configs`. Alongside cross-entropy
    loss, track accuracy and top-five accuracy on training and validation data.
    Rerun this cell before each new experiment to start with a fresh model.
    Rerunning only the training cell continues from the current model weights.
    """)
    return


@app.cell
def _(configs):
    tf.keras.utils.set_random_seed(configs["seed"])
    model = get_model(configs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=configs["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top@5_accuracy"),
        ],
    )
    model.summary()
    return (model,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Train

    Initialize Weights & Biases before constructing the callbacks, then pass
    them to `model.fit`:

    - `WandbMetricsLogger(log_freq=10)` logs training metrics every ten batches,
      plus epoch and validation metrics at the end of each epoch.
    - `WandbModelCheckpoint` saves the full model to `models/model.keras` at
      the end of each epoch and logs it as a Weights & Biases Artifact of type `model`.
      The `.keras` file includes the model architecture, weights, and optimizer
      state, so it can be loaded with `tf.keras.models.load_model`.

    The local file is overwritten each epoch. Weights & Biases versions the checkpoints
    under the run's model artifact, so you can inspect earlier versions.

    The context manager finishes the run after training, including if training
    raises an exception.
    """)
    return


@app.cell
def _(configs, entity, model, project, trainloader, validloader):
    os.makedirs("models", exist_ok=True)
    with wandb.init(entity=entity, project=project, config=configs):
        history = model.fit(
            trainloader,
            epochs=configs["epochs"],
            validation_data=validloader,
            shuffle=False,  # The tf.data training pipeline already shuffles.
            callbacks=[
                WandbMetricsLogger(log_freq=10),
                WandbModelCheckpoint(
                    filepath="models/model.keras",
                    save_freq="epoch",
                ),
            ],
        )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Explore your results

    Open the run link printed by Weights & Biases. In the run's **Artifacts**
    tab, find the artifact of type `model` and inspect its versions. Each version
    contains a `model.keras` checkpoint; the `latest` alias points to the latest
    version. In the **Workspace**, inspect `epoch/loss`, `epoch/val_loss`, and
    `epoch/val_accuracy`. Check **Config** for the hyperparameters used to
    produce these checkpoints.

    To keep a separate local checkpoint for each epoch, change `filepath` to
    `"models/model-{epoch:02d}.keras"`. Rebuild the model with the compile cell
    and train again, then compare the local files with the Weights & Biases Artifact versions.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Helper functions

    The input pipeline maps images and labels, shuffles only training examples,
    then batches and prefetches them.
    """)
    return


@app.function
def parse_data(example, num_classes):
    # Get image
    image = example["image"]
    image = tf.cast(image, tf.float32)

    # Get label
    label = example["label"]
    label = tf.one_hot(label, depth=num_classes)
    return image, label


@app.function
def get_dataloader(ds, configs, dataloader_type="train"):
    dataloader = ds.map(
        lambda example: parse_data(example, configs["num_classes"]),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if dataloader_type == "train":
        dataloader = dataloader.shuffle(configs["shuffle_buffer"], seed=configs["seed"])
    return dataloader.batch(configs["batch_size"]).prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    app.run()
