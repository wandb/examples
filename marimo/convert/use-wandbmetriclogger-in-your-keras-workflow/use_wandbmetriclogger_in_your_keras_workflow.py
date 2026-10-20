# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "importlib-resources==7.1.0",
#     "marimo>=0.23.11",
#     "tensorflow>=2.16,<3",
#     "tensorflow-datasets>=4.9,<5",
#     "wandb>=0.18,<1",
# ]
#
# [tool.marimo.runtime]
# on_cell_change = "lazy"
# ///

import marimo

__generated_with = "0.25.1"
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
    # Using WandbMetricsLogger in your Keras workflow

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/use-wandbmetriclogger-in-your-keras-workflow/use_wandbmetriclogger_in_your_keras_workflow.py/server)

    This notebook introduces the `WandbMetricsLogger` callback for [experiment tracking](https://docs.wandb.ai/models/integrations/keras). It logs training and validation metrics along with system metrics to Weights & Biases.

    Use Weights & Biases for machine learning experiment tracking, dataset
    versioning, and project collaboration.

    This notebook is designed to run on your local machine's CPU.

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

    Initialize Weights & Biases before constructing `WandbMetricsLogger`, then pass the
    callback to `model.fit`. `log_freq=10` logs training metrics every ten
    batches; epoch and validation metrics are also logged at epoch end. Use
    `log_freq="epoch"` for epoch-only logging or `"batch"` for every batch.

    The context manager finishes the run after training, including if training
    raises an exception.
    """)
    return


@app.cell
def _(configs, entity, model, project, trainloader, validloader):
    with wandb.init(entity=entity, project=project, config=configs):
        history = model.fit(
            trainloader,
            epochs=configs["epochs"],
            validation_data=validloader,
            shuffle=False,  # The tf.data training pipeline already shuffles.
            callbacks=[WandbMetricsLogger(log_freq=10)],
        )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Explore your results

    Open the run link printed by Weights & Biases. In the run's Workspace, inspect
    `epoch/loss`, `epoch/accuracy`, `epoch/val_loss`, and `epoch/val_accuracy`.
    Compare these epoch summaries with the more frequent `batch/*` metrics.
    Check Config for the batch size and learning rate recorded with the run.

    Try a different learning rate in `configs`, rebuild the model with the
    compile cell, and train again. Compare validation loss and accuracy across
    the two runs in your project. You can also change `log_freq` to see how
    logging frequency affects the detail in your training charts.
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
