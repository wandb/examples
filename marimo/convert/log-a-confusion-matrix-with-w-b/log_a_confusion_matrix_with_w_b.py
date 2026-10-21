# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "aiohttp>=3.11",
#     "fsspec==2026.7.0",
#     "marimo>=0.24.0",
#     "numpy>=1.26",
#     "pillow==12.3.0",
#     "scipy>=1.14.1",
#     "requests>=2.32",
#     "tensorflow==2.21.0",
#     "wandb==0.30.0",
# ]
# ///
"""Fine-tune InceptionV3 and log a confusion matrix to Weights & Biases."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    app_title="Plot a Confusion Matrix with Weights & Biases",
)

with app.setup:
    import shutil
    import tempfile
    import zipfile
    from pathlib import Path

    import fsspec
    import marimo as mo
    import numpy as np
    import PIL
    import tensorflow as tf
    import wandb
    from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
    from tensorflow.keras.callbacks import Callback
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

    DATASET_URL = "https://storage.googleapis.com/wandb_datasets/nature_12K.zip"
    RUNTIME_CACHE_ROOT = (
        Path(tempfile.gettempdir()) / "data" / "marimo"/ "inaturalist-12k"
    )


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

    # Plot a Confusion Matrix with Weights & Biases

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/log-a-confusion-matrix-with-w-b/log_a_confusion_matrix_with_w_b.py/server)

    Learn to log a confusion matrix after each training epoch and use it to
    identify which classes a model confuses. The example trains a classification
    head on a frozen InceptionV3 backbone using a ten-class sample from
    [iNaturalist 2017](https://github.com/visipedia/inat_comp).

    ![confusion_matrix](https://i.imgur.com/rvKx8RF.png)

    The small training defaults are intended to demonstrate logging. They do
    not establish an expected accuracy. Inspect the confusion matrix alongside
    validation loss and accuracy before deciding what to change.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Edit these values to compare runs. Training and validation use whole batches,
    rounded up to cover the requested example counts (up to the dataset size).
    Increase `num_train` or the epoch count to test whether additional training
    improves validation performance.
    The confusion matrix uses the first `num_log_batches` validation batches each
    epoch. Increase this value to include more examples in the chart.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Authenticate with Weights & Biases

    Paste your [Forge API key](https://docs.coreweave.com/products/wandb/platform/hosting/iam/api-keys) into the password field below. Keep your key out of the notebook’s source code.

    If this runtime already has credentials in `WANDB_API_KEY` or your `.netrc` file, leave the API key field blank.

    ## Set your team and project

    Enter your Weights & Biases team name and choose a project for this lesson. The default project is `confusion_matrix`. Then select **Connect to Weights & Biases**.
    """)
    return


@app.cell(hide_code=True)
def _():
    wandb_setup_form = mo.md("{api_key}\n\n{entity}\n\n{project}").batch(
        api_key=mo.ui.text(kind="password", label="Forge API key", full_width=True),
        entity=mo.ui.text(label="Team", placeholder="Your team name", full_width=True),
        project=mo.ui.text(value="confusion_matrix", label="Project", full_width=True),
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


@app.cell
def _():
    training_config = {
        "num_train": 100,
        "num_val": 500,
        "fc_size": 1024,
        "img_width": 299,
        "img_height": 299,
        "batch_size": 32,
        "pretrain_epochs": 1,
        "num_log_batches": 15,
        "random_seed": 23,
    }
    return (training_config,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Prepare the iNaturalist sample

    The [iNaturalist 2017](https://github.com/visipedia/inat_comp) sample contains
    10,000 training images and 2,000 validation images across 10 classes.
    **The archive is about 3.6 GB**, even when training on a small sample.
    The cells below expose the archive and define a download helper. The call
    to `ensure_dataset()` downloads and extracts it. Later cells load ImageNet
    weights and start training. Download and training time depend on your network
    and hardware.

    `dataset_fs` exposes the public archive in marimo's data side panel without
    downloading it. `ensure_dataset()` streams it into a temporary runtime cache
    and reuses the extracted directories on later runs.
    """)
    return


@app.cell
def _():
    dataset_fs = fsspec.filesystem(
        "reference",
        fo={"version": 1, "refs": {"nature_12K.zip": [DATASET_URL]}},
    )
    return (dataset_fs,)


@app.function
def ensure_dataset(dataset_fs):
    """Materialize and extract the remote dataset once, then return its paths."""
    archive_path = RUNTIME_CACHE_ROOT / "nature_12K.zip"
    train_path = RUNTIME_CACHE_ROOT / "inaturalist_12K" / "train"
    validation_path = RUNTIME_CACHE_ROOT / "inaturalist_12K" / "val"

    if train_path.is_dir() and validation_path.is_dir():
        return str(train_path), str(validation_path)

    RUNTIME_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        partial_path = archive_path.with_suffix(".zip.part")
        # Keras needs local directories, so stream the remote archive into the
        # runtime cache without loading 3.6 GB into memory.
        with dataset_fs.open("nature_12K.zip", "rb") as source:
            with partial_path.open("wb") as destination:
                shutil.copyfileobj(source, destination)
        partial_path.replace(archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(archive_path.parent)

    if not train_path.is_dir() or not validation_path.is_dir():
        raise FileNotFoundError(
            "The archive did not contain inaturalist_12K/train and "
            "inaturalist_12K/val."
        )

    return str(train_path), str(validation_path)


@app.cell
def _(dataset_fs):
    train_data, val_data = ensure_dataset(dataset_fs)
    return train_data, val_data


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Build the data generators and model

    InceptionV3 expects pixels scaled to `[-1, 1]`, so both generators use its
    `preprocess_input()` function. Only training images are augmented and shuffled.
    Validation keeps a fixed order, and class names follow the generator's numeric
    IDs so each row and column of the confusion matrix has the correct label.

    `build_model()` freezes the ImageNet backbone and trains a new classification
    head. Only the new head’s weights are updated in this example.
    """)
    return


@app.cell
def _(train_data, training_config, val_data):
    tf.keras.utils.set_random_seed(training_config["random_seed"])

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        train_data,
        target_size=(training_config["img_height"], training_config["img_width"]),
        batch_size=training_config["batch_size"],
        class_mode="categorical",
        seed=training_config["random_seed"],
    )

    class_names = list(train_generator.class_indices)

    val_generator = validation_datagen.flow_from_directory(
        val_data,
        classes=class_names,
        target_size=(training_config["img_height"], training_config["img_width"]),
        batch_size=training_config["batch_size"],
        class_mode="categorical",
        shuffle=False,
    )
    return class_names, train_generator, val_generator


@app.function
def build_model(fc_size, num_classes):
    """Load InceptionV3 with ImageNet weights, freeze it,
    and attach a finetuning top for this classification task.
    """
    # Load InceptionV3 as the base.
    base = InceptionV3(weights="imagenet", include_top=False)

    # Freeze the base layers.
    for layer in base.layers:
        layer.trainable = False

    x = base.get_layer("mixed10").output

    # Attach a fine-tuning layer.
    x = GlobalAveragePooling2D()(x)
    x = Dense(fc_size, activation="relu")(x)
    guesses = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=guesses)
    model.compile(
        optimizer="rmsprop",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Log a confusion matrix after each epoch

    The custom Keras callback collects validation images and their labels from
    the same batches, predicts class probabilities, and takes `argmax` to get
    class IDs. `wandb.plot.confusion_matrix()` builds a Weights & Biases chart from those IDs
    and the true labels. `training_run.log()` saves it under the stable key
    `conf_mat`, making it easy to compare epochs and runs. Diagonal entries are
    correct predictions. Off-diagonal entries show which classes are confused.
    The matrix covers the validation batches selected by `num_log_batches`,
    which may be fewer examples than the validation metrics cover.
    """)
    return


@app.class_definition
class ConfusionMatrixLogger(Callback):
    """Log predictions for a bounded validation sample after each epoch."""

    def __init__(self, training_run, generator, class_names, num_log_batches):
        super().__init__()
        if num_log_batches < 1:
            raise ValueError("num_log_batches must be at least 1")
        self.training_run = training_run
        self.generator = generator
        self.class_names = class_names
        self.num_batches = min(num_log_batches, len(generator))

    def on_epoch_end(self, epoch, logs=None):
        images, labels = zip(
            *(self.generator[index] for index in range(self.num_batches))
        )
        probabilities = self.model.predict(np.concatenate(images), verbose=0)
        self.training_run.log(
            {
                "epoch": epoch,
                "conf_mat": wandb.plot.confusion_matrix(
                    probs=None,
                    preds=probabilities.argmax(axis=1),
                    y_true=np.concatenate(labels).argmax(axis=1),
                    class_names=self.class_names,
                ),
            }
        )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Train and save the results to Weights & Biases

    `wandb.init()` records the experiment configuration. `WandbMetricsLogger`
    logs loss and accuracy, `WandbModelCheckpoint` saves a model Artifact each
    epoch, and `ConfusionMatrixLogger` logs the validation chart.

    The temporary checkpoint directory stays alive until the run finishes and
    Weights & Biases flushes the Artifact. The run context finishes automatically, including
    when training raises an error.
    """)
    return


@app.cell
def _(
    class_names,
    entity,
    project,
    train_generator,
    training_config,
    val_generator,
):
    with tempfile.TemporaryDirectory(
        prefix="confusion-matrix-"
    ) as checkpoint_directory:
        with wandb.init(
            project=project,
            entity=entity,
            name="inaturalist-inceptionv3-confusion-matrix",
            config={**training_config, "num_classes": len(class_names)},
        ) as training_run:
            model = build_model(training_config["fc_size"], len(class_names))
            callbacks = [
                WandbMetricsLogger(),
                WandbModelCheckpoint(
                    str(Path(checkpoint_directory) / "checkpoint.keras")
                ),
                ConfusionMatrixLogger(
                    training_run,
                    val_generator,
                    class_names,
                    training_config["num_log_batches"],
                ),
            ]
            history = model.fit(
                train_generator,
                steps_per_epoch=min(
                    len(train_generator),
                    max(1, int(np.ceil(
                        training_config["num_train"] / training_config["batch_size"]
                    ))),
                ),
                epochs=training_config["pretrain_epochs"],
                validation_data=val_generator,
                validation_steps=min(
                    len(val_generator),
                    max(1, int(np.ceil(
                        training_config["num_val"] / training_config["batch_size"]
                    ))),
                ),
                callbacks=callbacks,
            )
            run_url = training_run.url     
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Open the run in Weights & Biases App UI and find `conf_mat` in the **Custom Charts** panel.
    Compare the mistakes across classes, then increase the training sample or
    epoch count and run again.

    ## Next steps

    - More info and customization details: [Confusion Matrix](https://wandb.ai/wandb/plots/reports/Confusion-Matrix--VmlldzozMDg1NTM)
    - More examples in this Weights & Biases project: [Custom Charts](https://app.wandb.ai/demo-team/custom-charts).
    """)
    return


if __name__ == "__main__":
    app.run()
