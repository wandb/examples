# /// script
# dependencies = [
#     "tensorflow==2.21.0",
#     "tensorflow-datasets==4.9.10",
#     "wandb==0.30.0",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(auto_download=["html"])

with app.setup:
    import marimo as mo
    import os
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras import models
    import tensorflow_datasets as tfds

    # Weights & Biases integrations
    import wandb
    from wandb.integration.keras import WandbMetricsLogger
    from wandb.integration.keras import WandbModelCheckpoint


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Using Keras Checkpoint callback with Weights & Biases

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/use-wandbmodelcheckpoint-in-your-keras-workflow/use_wandbmodelcheckpoint_in_your_keras_workflow.py/server)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This notebook introduces the `WandbModelCheckpoint` callback. Use this callback to log your model checkpoints to Weights & Biases [Artifacts](https://docs.wandb.ai/guides/data-and-model-versioning).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <style>
    .wandb-by-cw-logo--dark {
      display: none;
    }

    :host-context(body.dark) .wandb-by-cw-logo--light {
      display: none;
    }

    :host-context(body.dark) .wandb-by-cw-logo--dark {
      display: block;
    }
    </style>

    ## Why should I use W&B?

    <img class="wandb-by-cw-logo--light" src="https://raw.githubusercontent.com/wandb/docs/main/icons/Endorsed_primary_blackwhite.svg" width="320" alt="Weights & Biases by CoreWeave" />
    <img class="wandb-by-cw-logo--dark" src="https://raw.githubusercontent.com/wandb/docs/main/icons/Endorsed_primary_goldwhite.svg" width="320" alt="Weights & Biases by CoreWeave" />

    Use Weights & Biases for machine learning experiment tracking, dataset versioning, and project collaboration.

    <img src="https://wandb.me/mini-diagram" width="650" alt="Weights & Biases features" />
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Authentication

    Enter your [W&B API key](https://wandb.ai/authorize) and, if needed, your team or entity. You can leave the key blank when this environment already has W&B credentials. If this is your first time using W&B, [create a free account](https://wandb.ai/signup).
    """)
    return


@app.cell(hide_code=True)
def _():
    _api_key_input = mo.ui.text(
        kind="password",
        label="W&B API key (optional)",
        placeholder="Paste a key or use cached credentials",
        full_width=True,
    )
    _entity_input = mo.ui.text(
        label="W&B entity or team (optional)",
        placeholder="Leave blank to use your default entity",
        full_width=True,
    )
    wandb_login_form = (
        mo.md("{api_key}\n\n{entity}")
        .batch(api_key=_api_key_input, entity=_entity_input)
        .form(submit_button_label="Connect to W&B", bordered=True)
    )
    wandb_login_form
    return (wandb_login_form,)


@app.cell(hide_code=True)
def _(wandb_login_form):
    mo.stop(
        wandb_login_form.value is None,
        mo.callout(
            mo.md("Connect to W&B above before preparing data or training the model."),
            kind="info",
        ),
    )

    _api_key = wandb_login_form.value["api_key"].strip()
    _entity = wandb_login_form.value["entity"].strip()
    try:
        _login_ok = wandb.login(key=_api_key or None, relogin=bool(_api_key))
        _login_error = None
    except wandb.errors.Error as _error:
        _login_ok = False
        _login_error = str(_error)

    mo.stop(
        not _login_ok,
        mo.callout(
            mo.md(
                "W&B authentication did not complete. "
                f"Check the API key and try again.\n\nW&B reported: `{_login_error or 'unknown error'}`"
            ),
            kind="danger",
        ),
    )

    wandb_settings = {
        "project": "intro-keras",
        "entity": _entity or None,
    }
    mo.callout(mo.md("Connected to W&B."), kind="success")
    return (wandb_settings,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Hyperparameters

    Use of a proper config system is a recommended best practice for reproducible machine learning. We can track the hyperparameters for every experiment using W&B. In this notebook we use a simple Python `dict` as our config system.

    Choose the training values below. Submitting the form downloads Fashion-MNIST and ImageNet weights, starts a W&B run, trains the model, and logs model checkpoints.
    """)
    return


@app.cell(hide_code=True)
def _():
    _epochs_input = mo.ui.number(
        start=1,
        stop=50,
        step=1,
        value=10,
        label="Epochs",
    )
    _batch_size_input = mo.ui.number(
        start=16,
        stop=256,
        step=16,
        value=64,
        label="Batch size",
    )
    training_form = (
        mo.md("{epochs}\n\n{batch_size}")
        .batch(epochs=_epochs_input, batch_size=_batch_size_input)
        .form(submit_button_label="Download data and train with W&B", bordered=True)
    )
    training_form
    return (training_form,)


@app.cell(hide_code=True)
def _(training_form, wandb_settings):
    mo.stop(
        training_form.value is None,
        mo.callout(
            mo.md("Choose the hyperparameters and submit the form to continue."),
            kind="info",
        ),
    )
    mo.stop(
        not wandb_settings,
        mo.callout(mo.md("Connect to W&B before training."), kind="info"),
    )
    training_request = dict(training_form.value)
    return (training_request,)


@app.cell
def _(training_request):
    configs = dict(
        dataset="fashion_mnist",
        num_classes=10,
        shuffle_buffer=1024,
        batch_size=int(training_request["batch_size"]),
        image_size=28,
        image_channels=1,
        earlystopping_patience=3,
        learning_rate=1e-3,
        epochs=int(training_request["epochs"]),
    )
    return (configs,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Dataset

    In this notebook, we use the [Fashion-MNIST](https://www.tensorflow.org/datasets/catalog/fashion_mnist) dataset from the TensorFlow Datasets catalog. We aim to build a simple image classification pipeline using TensorFlow/Keras.
    """)
    return


@app.cell
def _(configs):
    train_ds, valid_ds = tfds.load(
        configs["dataset"],
        split=["train", "test"],
    )
    return train_ds, valid_ds


@app.cell
def _(configs):
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
def _():
    mo.md(r"""
    ## Model
    """)
    return


@app.function
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


@app.cell
def _(configs):
    tf.keras.backend.clear_session()
    model = get_model(configs)
    model.summary()
    return (model,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Compile model
    """)
    return


@app.cell
def _(model):
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top@5_accuracy"),
        ],
    )
    compiled_model = model
    return (compiled_model,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Train

    The submitted action creates a W&B run, trains the classifier, logs its metrics, and saves each model checkpoint to W&B Artifacts.
    """)
    return


@app.cell
def _(compiled_model, configs, trainloader, validloader, wandb_settings):
    os.makedirs("models", exist_ok=True)
    if wandb.run is not None:
        wandb.finish()

    # Initialize a W&B run
    with wandb.init(
        project=wandb_settings["project"],
        entity=wandb_settings["entity"],
        config=configs,
    ) as _run:
        wandb_run_url = _run.url
        mo.output.replace(
            mo.callout(
                mo.md(
                    f"[**Open this training run in W&B**]({wandb_run_url})\n\n"
                    "Metrics will stream to W&B while the model trains."
                ),
                kind="info",
                title="Training in progress",
            )
        )

        # Train your model
        training_history = compiled_model.fit(
            trainloader,
            epochs=configs["epochs"],
            validation_data=validloader,
            callbacks=[
                WandbMetricsLogger(log_freq=10),
                # Notice the use of WandbModelCheckpoint here
                WandbModelCheckpoint(filepath="models/model.keras"),
            ],
        )

    mo.output.replace(None)
    return (wandb_run_url,)


@app.cell(hide_code=True)
def _(wandb_run_url):
    mo.callout(
        mo.md(f"""
    ### Training complete

    [**Open this training run in W&B**]({wandb_run_url})

    Open the run's **Artifacts** tab to inspect the model checkpoints logged by `WandbModelCheckpoint`.
    """),
        kind="success",
    )
    return


if __name__ == "__main__":
    app.run()
