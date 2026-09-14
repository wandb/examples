# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.23.11",
#     "tensorflow>=2.16,<3",
#     "tensorflow-datasets>=4.9,<5",
#     "wandb>=0.18,<1",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="Keras metrics with W&B")

with app.setup:
    import marimo as mo
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from tensorflow.keras import layers, models

    # Weights and Biases related imports
    import wandb
    from wandb.integration.keras import WandbMetricsLogger


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Using Keras MetricsLogger in your Keras workflow

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/use-wandbmetriclogger-in-your-keras-workflow/use_wandbmetriclogger_in_your_keras_workflow.py/server)

    This notebook introduces the `WandbMetricsLogger` callback for [experiment tracking](https://docs.wandb.ai/models/integrations/keras). It logs training and validation metrics along with system metrics to Weights & Biases.

    Use Weights & Biases for machine learning experiment tracking, dataset
    versioning, and project collaboration. See the
    [original Colab tutorial](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Use_WandbMetricLogger_in_your_Keras_workflow.ipynb).

    ## Setup and installation

    Open this notebook with `uvx marimo edit use_wandbmetriclogger_in_your_keras_workflow.py --sandbox` to install
    its dependencies. A CPU is sufficient; a GPU speeds up training. The first
    training submission downloads Fashion-MNIST and ImageNet weights.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Authentication

    Enter your [W&B API key](https://wandb.ai/authorize), or leave it blank to use
    `WANDB_API_KEY` from the marimo Secrets panel or credentials already stored
    in this runtime. A fresh molab session does not inherit your local login.
    If needed, [create a free W&B account](https://wandb.ai/signup).

    The entity is the team name in your W&B project URL:
    `wandb.ai/<entity>/<project>`. Leave it blank to use your default entity.
    A blank run name lets W&B generate a name for each experiment.

    Submit the form to authenticate, download data and weights, create one W&B
    run, and train the model. The callback streams training and validation metrics to that run.
    """)
    return


@app.cell(hide_code=True)
def _():
    training_form = (
        mo.md(
            """
            {api_key}

            {entity}

            {project}

            {run_name}

            ## Hyperparameters

            A Python dictionary records these hyperparameters in the W&B run
            config so experiments can be compared and reproduced.

            {epochs}  {batch_size}

            {learning_rate}
            """
        )
        .batch(
            api_key=mo.ui.text(kind="password", label="W&B API key (optional)", full_width=True),
            entity=mo.ui.text(label="W&B entity or team (optional)", full_width=True),
            project=mo.ui.text(value="intro-keras", label="W&B project", full_width=True),
            run_name=mo.ui.text(label="Run name (blank auto-generates)", full_width=True),
            epochs=mo.ui.number(start=1, stop=50, step=1, value=10, label="Epochs"),
            batch_size=mo.ui.dropdown(options=[32, 64, 128, 256], value=64, label="Batch size"),
            learning_rate=mo.ui.number(start=0.00001, stop=0.1, step=0.00001, value=0.001, label="Learning rate"),
        )
        .form(submit_button_label="Download data and train with W&B", bordered=True)
    )
    training_form
    return (training_form,)


@app.cell(hide_code=True)
def _(training_form):
    mo.stop(
        training_form.value is None,
        mo.callout(mo.md("Submit the form above to download data and train."), kind="info"),
    )
    _submitted = training_form.value
    mo.stop(
        not _submitted["project"].strip(),
        mo.callout(mo.md("Enter a W&B project name and submit again."), kind="danger"),
    )
    _api_key = _submitted["api_key"].strip()
    try:
        _login_ok = wandb.login(key=_api_key or None, relogin=bool(_api_key))
    except (wandb.errors.Error, ValueError):
        _login_ok = False
    mo.stop(
        not _login_ok,
        mo.callout(
            mo.md("W&B authentication did not complete. Check your API key or runtime secret, then submit again."),
            kind="danger",
        ),
    )
    wandb_settings = {
        "project": _submitted["project"].strip(),
        "entity": _submitted["entity"].strip() or None,
        "name": _submitted["run_name"].strip() or None,
    }
    training_request = {
        "epochs": int(_submitted["epochs"]),
        "batch_size": int(_submitted["batch_size"]),
        "learning_rate": float(_submitted["learning_rate"]),
    }
    mo.callout(mo.md("Authenticated. Preparing the submitted training run."), kind="success")
    return training_request, wandb_settings


@app.cell
def _(training_request):
    configs = dict(
        dataset="fashion_mnist",
        num_classes=10,
        shuffle_buffer=1024,
        batch_size=training_request["batch_size"],
        image_size=28,
        image_channels=1,
        learning_rate=training_request["learning_rate"],
        epochs=training_request["epochs"],
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
    The same cell creates and compiles the model so training always receives a
    fresh, compiled model after a new submission.
    """)
    return


@app.cell
def _(configs):
    tf.keras.backend.clear_session()
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

    Initialize W&B before constructing `WandbMetricsLogger`, then pass the
    callback to `model.fit`. `log_freq=10` logs training metrics every ten
    batches; epoch and validation metrics are also logged at epoch end. Use
    `log_freq="epoch"` for epoch-only logging or `"batch"` for every batch.

    The context manager finishes the run after training, including if training
    raises an exception.
    """)
    return


@app.cell
def _(configs, model, trainloader, validloader, wandb_settings):
    if wandb.run is not None:
        wandb.run.finish()

    # Initialize a W&B run
    with wandb.init(**wandb_settings, config=configs) as training_run:
        wandb_run_url = training_run.url
        # Train your model
        training_history = model.fit(
            trainloader,
            epochs=configs["epochs"],
            validation_data=validloader,
            shuffle=False,  # The tf.data training pipeline already shuffles.
            callbacks=[WandbMetricsLogger(log_freq=10)],  # Notice the use of WandbMetricsLogger here
        )
        # Close the W&B run
    training_result = {
        "url": wandb_run_url,
        "epochs": len(training_history.history["loss"]),
    }
    return (training_result,)


@app.cell(hide_code=True)
def _(training_result):
    mo.callout(
        mo.md(
            f"Training finished after {training_result['epochs']} epochs. "
            f"[Open the W&B run]({training_result['url']}).\n\n"
            "In the run's Workspace, inspect `epoch/loss`, `epoch/accuracy`, "
            "`epoch/val_loss`, and `epoch/val_accuracy`, then compare the "
            "`batch/*` charts. Check Config for the batch size and learning rate."
        ),
        kind="success",
    )
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
    # image = tf.image.convert_image_dtype(image, dtype=tf.float32)
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
