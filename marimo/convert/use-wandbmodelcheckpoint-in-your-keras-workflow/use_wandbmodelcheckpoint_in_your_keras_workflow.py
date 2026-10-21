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
app = marimo.App(
    width="medium",
    app_title="Keras checkpoints with Weights & Biases",
)

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
    <style>
    .wandb-config-logo--dark {
      display: none;
    }

    body.dark .wandb-config-logo--light {
      display: none;
    }

    body.dark .wandb-config-logo--dark {
      display: block;
    }
    </style>

    <img class="wandb-config-logo--light" src="https://raw.githubusercontent.com/wandb/docs/main/icons/Endorsed_primary_blackwhite.svg" width="400" alt="Weights & Biases by CoreWeave" />
    <img class="wandb-config-logo--dark" src="https://raw.githubusercontent.com/wandb/docs/main/icons/Endorsed_primary_goldwhite.svg" width="400" alt="Weights & Biases by CoreWeave" />

    # Using WandbModelCheckpoint in your Keras workflow

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/use-wandbmodelcheckpoint-in-your-keras-workflow/use_wandbmodelcheckpoint_in_your_keras_workflow.py/server)

    This notebook introduces the `WandbModelCheckpoint` callback. It saves Keras
    model checkpoints and logs them as versioned Weights & Biases [Artifacts](https://docs.wandb.ai/models/artifacts).
    Use `WandbMetricsLogger` alongside it to track training and validation metrics.

    Use Weights & Biases for machine learning experiment tracking, dataset
    versioning, and project collaboration.

    This notebook is designed to run on your local machine's CPU. The data
    and model cells download Fashion-MNIST and ImageNet weights the first time.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Authenticate with Weights & Biases

    Paste your [Forge API key](https://docs.coreweave.com/products/wandb/platform/hosting/iam/api-keys) into the password field below. Keep your key out of the notebook’s source code.

    If this runtime already has credentials in `WANDB_API_KEY` or your `.netrc` file, leave the API key field blank.

    ## Set your team and project

    Enter your Weights & Biases team name and choose a project for this lesson. The default project is `intro-keras`. Then select **Connect to Weights & Biases**.
    """)
    return


@app.cell(hide_code=True)
def _():
    wandb_setup_form = mo.md("{api_key}\n\n{entity}\n\n{project}").batch(
        api_key=mo.ui.text(kind="password", label="Forge API key", full_width=True),
        entity=mo.ui.text(label="Team", placeholder="Your team name", full_width=True),
        project=mo.ui.text(value="intro-keras", label="Project", full_width=True),
    ).form(submit_button_label="Connect to Weights & Biases", bordered=True)
    wandb_setup_form
    return (wandb_setup_form,)


@app.cell(hide_code=True)
def _(wandb_setup_form):
    mo.stop(wandb_setup_form.value is None)

    _submission = wandb_setup_form.value
    _api_key = _submission["api_key"].strip()
    _entity = _submission["entity"].strip()
    _project = _submission["project"].strip()
    mo.stop(
        not _entity or not _project,
        mo.callout("Enter your team and project, then connect again.", kind="warn"),
    )

    try:
        _login_ok = wandb.login(key=_api_key or None, relogin=bool(_api_key))
    except (wandb.errors.Error, ValueError):
        _login_ok = False

    mo.stop(
        not _login_ok,
        mo.callout("Check your API key or saved credentials, then connect again.", kind="danger"),
    )

    entity = _entity
    project = _project
    mo.md("✓ Connected to Weights & Biases.")
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
