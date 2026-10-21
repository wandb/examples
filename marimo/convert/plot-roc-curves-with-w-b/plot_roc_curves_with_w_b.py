# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "marimo>=0.24.2",
#     "numpy>=1.26",
#     "pandas>=2.2",
#     "pillow==12.3.0",
#     "scikit-learn>=1.5",
#     "scipy>=1.14.1",
#     "tensorflow==2.21.0",
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
    app_title="Plot ROC Curves with Weights & Biases",
)

with app.setup:
    import shutil
    import tempfile
    import zipfile
    from pathlib import Path
    from urllib.request import urlopen

    import marimo as mo
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import wandb
    from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
    from tensorflow.keras.callbacks import Callback
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from wandb.integration.keras import WandbMetricsLogger

    DATASET_URL = "https://storage.googleapis.com/wandb_datasets/nature_12K.zip"
    RUNTIME_CACHE_ROOT = (
        Path(tempfile.gettempdir()) / "wandb-examples" / "inaturalist-12k"
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

    # Plot ROC Curves with Weights & Biases

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/plot-roc-curves-with-w-b/plot_roc_curves_with_w_b.py/server)

    Learn to log multiclass ROC curves with `wandb.plot.roc_curve()`. This lesson
    trains a classification head on a frozen, ImageNet-trained InceptionV3 model
    to identify ten groups of living things from an iNaturalist image sample.

    An ROC curve plots the true-positive rate against the false-positive rate
    as the decision threshold changes. For multiclass predictions, each curve
    treats one class as positive and all other classes as negative.

    ![Example ROC curves in Weights & Biases](https://i.imgur.com/CqGXSzj.png)

    The small training defaults let you try the workflow on a CPU. More data
    and epochs can improve the curves; a GPU speeds up training.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Authenticate with Weights & Biases

    Paste your [Forge API key](https://docs.coreweave.com/products/wandb/platform/hosting/iam/api-keys) into the password field below. Keep your key out of the notebook’s source code.

    If this runtime already has credentials in `WANDB_API_KEY` or your `.netrc` file, leave the API key field blank.

    ## Set your team and project

    Enter your Weights & Biases team name and choose a project for this lesson. The default project is `custom_roc_curve`. Then select **Connect to Weights & Biases**.
    """)
    return


@app.cell(hide_code=True)
def _():
    wandb_setup_form = mo.md("{api_key}\n\n{entity}\n\n{project}").batch(
        api_key=mo.ui.text(kind="password", label="Forge API key", full_width=True),
        entity=mo.ui.text(label="Team", placeholder="Your team name", full_width=True),
        project=mo.ui.text(value="custom_roc_curve", label="Project", full_width=True),
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
    ## Experiment configuration

    Start with 100 training images, 500 validation images, and one epoch. We
    sample evenly across the ten classes so every class has positive and
    negative examples for its ROC curve. Keep the seed and validation size
    fixed when comparing experiments.

    Passing this dictionary to `wandb.init(config=config)` records the settings
    with the run. Try increasing `num_train` or `epochs` after your first run.
    """)
    return


@app.cell
def _():
    config = {
        "num_train": 100,
        "num_val": 500,
        "num_classes": 10,
        "fc_size": 1024,
        "img_width": 299,
        "img_height": 299,
        "batch_size": 32,
        "epochs": 1,
        "random_seed": 23,
    }
    return (config,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Prepare the dataset

    The [iNaturalist sample](https://github.com/visipedia/inat_comp) contains
    10,000 training images and 2,000 validation images across ten classes.
    **The first run downloads and extracts an approximately 3.6 GB archive,**
    even with the small training defaults. Later runs reuse the temporary
    runtime cache.

    `sample_images` selects the requested number of files from each split,
    distributing them across classes. The download and sampling helpers are
    defined at the end of the notebook.
    """)
    return


@app.cell
def _(config):
    train_path, validation_path = ensure_dataset()
    train_samples = sample_images(train_path, config["num_train"], config["random_seed"])
    validation_samples = sample_images(
        validation_path, config["num_val"], config["random_seed"]
    )
    class_names = sorted(train_samples["class"].unique())
    return class_names, train_samples, validation_samples


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Prepare image batches

    Augment the training images with shear, zoom, and horizontal flips. Both
    generators use InceptionV3's [preprocessing function](https://keras.io/api/applications/inceptionv3/inception_v3_preprocessing/),
    which scales pixel values to the range −1 to 1.

    Validation images are neither augmented nor shuffled. The same selected
    images, class IDs, and class order are used for validation and ROC logging
    at every epoch.
    """)
    return


@app.cell
def _(class_names, config, train_samples, validation_samples):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_dataframe(
        train_samples,
        x_col="filename",
        y_col="class",
        classes=class_names,
        target_size=(config["img_height"], config["img_width"]),
        batch_size=config["batch_size"],
        class_mode="categorical",
        shuffle=True,
        seed=config["random_seed"],
    )
    validation_generator = validation_datagen.flow_from_dataframe(
        validation_samples,
        x_col="filename",
        y_col="class",
        classes=class_names,
        target_size=(config["img_height"], config["img_width"]),
        batch_size=config["batch_size"],
        class_mode="categorical",
        shuffle=False,
    )
    return train_generator, validation_generator


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Build and compile the model

    Freeze the pretrained InceptionV3 base and train a new classification head.
    The final softmax layer produces one probability per class. Rerun the model
    creation cell before each new experiment to start with a fresh head;
    rerunning only the training cell continues from the current model weights.
    """)
    return


@app.function
def build_model(fc_size, num_classes):
    base = InceptionV3(weights="imagenet", include_top=False)
    base.trainable = False
    features = GlobalAveragePooling2D()(base.output)
    features = Dense(fc_size, activation="relu")(features)
    probabilities = Dense(num_classes, activation="softmax")(features)
    return Model(inputs=base.input, outputs=probabilities)


@app.cell
def _(config):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config["random_seed"])
    model = build_model(config["fc_size"], config["num_classes"])
    model.compile(
        optimizer="rmsprop",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return (model,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Log ROC curves at each epoch

    The callback passes three values to [`wandb.plot.roc_curve`](https://docs.wandb.ai/models/ref/python/custom-charts/roc_curve):

    - Ground-truth class IDs, with shape `(num_images,)`.
    - Predicted probabilities, with shape `(num_images, num_classes)`.
    - Class names in the same order as the probability columns.

    Use the probabilities rather than the predicted class IDs: ROC curves
    evaluate different thresholds. The callback processes one batch of images
    at a time and collects only labels and probabilities for the chart.

    Keep the `roc_curve` logging key consistent so runs can share a chart panel.
    """)
    return


@app.class_definition
class ROCCurveLogger(Callback):
    def __init__(self, run, generator):
        super().__init__()
        self.run = run
        self.generator = generator
        self.class_names = sorted(
            generator.class_indices, key=generator.class_indices.get
        )

    def on_epoch_end(self, epoch, logs=None):
        class_ids = []
        probabilities = []
        for index in range(len(self.generator)):
            images, labels = self.generator[index]
            class_ids.append(labels.argmax(axis=1))
            probabilities.append(self.model.predict_on_batch(images))

        self.run.log(
            {
                "roc_curve": wandb.plot.roc_curve(
                    np.concatenate(class_ids),
                    np.concatenate(probabilities),
                    labels=self.class_names,
                )
            }
        )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Train

    Create a Weights & Biases run, then pass both callbacks to `model.fit`.
    `WandbMetricsLogger` records loss and accuracy; `ROCCurveLogger` records
    per-class ROC curves. Training and validation use all of their selected
    images, including the final partial batch.

    The context manager finishes the run when training ends.
    """)
    return


@app.cell
def _(config, entity, model, project, train_generator, validation_generator):
    with wandb.init(entity=entity, project=project, config=config) as run:
        history = model.fit(
            train_generator,
            epochs=config["epochs"],
            validation_data=validation_generator,
            callbacks=[
                WandbMetricsLogger(),
                ROCCurveLogger(run, validation_generator),
            ],
        )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Explore your results

    Open the run link printed by Weights & Biases and find the `roc_curve`
    custom chart. Check that it contains all ten class names. Compare curves
    across epochs and runs: a curve closer to the upper-left corner indicates
    better separation of that class from the others.

    Inspect `epoch/loss`, `epoch/val_loss`, and `epoch/val_accuracy` alongside
    the curves. Increase `num_train` or `epochs`, rebuild the model, and train
    again while keeping the validation sample fixed.

    See [ROC chart customization](https://wandb.ai/wandb/plots/reports/Plot-ROC-Curves--VmlldzoyNjk3MDE)
    and the [Custom Charts project](https://app.wandb.ai/demo-team/custom-charts)
    for more examples.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Helper functions

    Cache the archive and extracted directories, then select a reproducible
    sample of images from each class. Use at least ten images per split to
    include every class, up to 10,000 for training and 2,000 for validation.
    """)
    return


@app.function
def ensure_dataset():
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
        with urlopen(DATASET_URL, timeout=60) as source:
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


@app.function
def sample_images(directory, num_images, seed):
    class_dirs = sorted(path for path in Path(directory).iterdir() if path.is_dir())
    if num_images < len(class_dirs):
        raise ValueError("Choose at least one image per class.")
    per_class, remainder = divmod(num_images, len(class_dirs))
    samples = []
    for index, class_dir in enumerate(class_dirs):
        filenames = sorted(
            str(path)
            for path in class_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        class_images = pd.DataFrame({"filename": filenames, "class": class_dir.name})
        samples.append(
            class_images.sample(n=per_class + (index < remainder), random_state=seed)
        )
    return pd.concat(samples, ignore_index=True)


if __name__ == "__main__":
    app.run()
