# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "fsspec[http]",
#     "marimo>=0.24.0",
#     "numpy>=1.26",
#     "pandas>=2.2",
#     "pillow==12.3.0",
#     "scikit-learn>=1.5",
#     "scipy>=1.14.1",
#     "tensorflow==2.21.0",
#     "wandb==0.30.0",
# ]
# ///
"""Fine-tune InceptionV3 and log multiclass ROC curves to W&B."""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="Plot ROC Curves with W&B")

with app.setup:
    import shutil
    import tempfile
    import zipfile
    from pathlib import Path

    import fsspec
    import marimo as mo
    import numpy as np
    import tensorflow as tf
    import wandb
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.callbacks import Callback
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from wandb.integration.keras import WandbMetricsLogger

    DATASET_URL = "https://storage.googleapis.com/wandb_datasets/nature_12K.zip"
    PROJECT_NAME = "custom_roc_curve"
    RUNTIME_CACHE_ROOT = (
        Path(tempfile.gettempdir()) / "wandb-examples" / "inaturalist-12k"
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Plot ROC Curves with W&B

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/plot-roc-curves-with-w-b/plot_roc_curves_with_w_b.py/server)

    Learn how to log [receiver operating characteristic (ROC) curves](https://scikit-learn.org/stable/modules/model_evaluation.html#receiver-operating-characteristic-roc) backed by [Vega](https://vega.github.io/vega/docs/) in [Weights & Biases](https://wandb.ai). This example fine-tunes InceptionV3 with ImageNet weights to identify 10 types of living things from the [iNaturalist 2017 dataset](https://github.com/visipedia/inat_comp).

    These simple cases explain the basics; W&B's query editor supports more sophisticated custom charts.

    ![ROC curves in W&B](https://i.imgur.com/CqGXSzj.png)

    The defaults intentionally use few training examples and one epoch so you can exercise the workflow. The original tutorial reports validation accuracy in the low 80s after roughly one epoch on the complete training dataset.

    ## Method: `wandb.plot.roc_curve()`

    - Read about [ROC curve customization](https://wandb.ai/wandb/plots/reports/Plot-ROC-Curves--VmlldzoyNjk3MDE).
    - Explore more examples in the [Custom Charts project](https://app.wandb.ai/demo-team/custom-charts).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Authentication

    Enter a [W&B API key](https://wandb.ai/authorize) and an optional team entity below. Leave the key blank when `WANDB_API_KEY` is configured in the Molab Secrets panel or this runtime already has W&B credentials. If needed, [create a free W&B account](https://wandb.ai/signup).

    Your entity is the first path segment after `wandb.ai` when you open a W&B workspace. The API key is used only for `wandb.login()` and is never included in the run configuration or notebook output.
    """)
    return


@app.cell(hide_code=True)
def _():
    _api_key_input = mo.ui.text(
        kind="password",
        label="W&B API key (optional)",
        placeholder="Paste a key or use configured credentials",
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
            mo.md("Connect to W&B before configuring the experiment."),
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
                "W&B authentication did not complete. Check the API key or "
                "Molab Secrets configuration and submit again.\n\n"
                f"W&B reported: `{_login_error or 'unknown error'}`"
            ),
            kind="danger",
        ),
    )

    wandb_settings = {
        "project": PROJECT_NAME,
        "entity": _entity or None,
    }
    _entity_note = (
        f" Runs will be logged to `{wandb_settings['entity']}`."
        if wandb_settings["entity"]
        else " Runs will use your default W&B entity."
    )
    mo.callout(mo.md(f"Connected to W&B.{_entity_note}"), kind="success")
    return (wandb_settings,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Setup: Access the remote dataset

    The iNaturalist sample contains 10,000 training images and 2,000 validation images, evenly distributed across 10 classes such as birds, insects, plants, and mammals. **The archive is approximately 3.6 GB, so the first transfer and extraction can take several minutes.**

    The `dataset_fs` value below exposes the public archive through an `fsspec` reference filesystem, so you can browse it in marimo's data side panel without transferring the whole file.

    Training still requires local image directories. The first time you submit the training form, `ensure_dataset()` streams the archive into a temporary runtime cache and extracts it there. Later submissions reuse that cache; no dataset files are written beside the notebook.
    """)
    return


@app.cell
def _():
    # Expose the remote dataset as a named filesystem so it appears in
    # marimo's data side panel.
    dataset_fs = fsspec.filesystem(
        "reference",
        fo={
            "version": 1,
            "refs": {"nature_12K.zip": [DATASET_URL]},
        },
    )
    return (dataset_fs,)


@app.function
def ensure_dataset(dataset_fs):
    """Materialize and extract the remote dataset once, then return its paths."""
    archive_path = RUNTIME_CACHE_ROOT / "nature_12K.zip"
    train_path = RUNTIME_CACHE_ROOT / "inaturalist_12K" / "train"
    validation_path = RUNTIME_CACHE_ROOT / "inaturalist_12K" / "val"

    if train_path.is_dir() and validation_path.is_dir():
        return str(train_path), str(validation_path), "Reused the extracted dataset."

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

    return str(train_path), str(validation_path), "Streamed and extracted the remote dataset."


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Training code

    The model freezes the ImageNet-trained InceptionV3 base and trains a new classification head. At the end of each epoch, `ROCCurveLogger` predicts several validation batches and sends the ground-truth class IDs and per-class probabilities to `wandb.plot.roc_curve()`.

    Change the number of training examples or epochs in the form to compare ROC curves. The small defaults favor demo speed over accuracy.
    """)
    return


@app.function
def build_model(fc_size, num_classes):
    """Freeze an ImageNet-trained InceptionV3 base and add a classifier head."""
    base = InceptionV3(weights="imagenet", include_top=False)
    for layer in base.layers:
        layer.trainable = False

    features = base.get_layer("mixed10").output
    features = GlobalAveragePooling2D()(features)
    features = Dense(fc_size, activation="relu")(features)
    predictions = Dense(num_classes, activation="softmax")(features)

    model = Model(inputs=base.input, outputs=predictions)
    model.compile(
        optimizer="rmsprop",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


@app.function
def make_roc_curve_logger(run, generator, num_log_batches):
    """Create a Keras callback that logs multiclass ROC curves each epoch."""

    class ROCCurveLogger(Callback):
        def __init__(self):
            super().__init__()
            self.generator = generator
            self.num_batches = num_log_batches
            self.class_names = [
                name
                for name, _class_id in sorted(
                    generator.class_indices.items(), key=lambda item: item[1]
                )
            ]

        def on_epoch_end(self, epoch, logs=None):
            validation_batches = [
                self.generator[index] for index in range(self.num_batches)
            ]
            validation_data = np.vstack(
                [batch_data for batch_data, _batch_labels in validation_batches]
            )
            validation_labels = np.vstack(
                [batch_labels for _batch_data, batch_labels in validation_batches]
            )

            validation_probabilities = self.model.predict(
                validation_data, verbose=0
            )
            ground_truth_class_ids = validation_labels.argmax(axis=1)

            # Keep this key stable so curves from multiple runs share a panel.
            run.log(
                {
                    "roc_curve": wandb.plot.roc_curve(
                        ground_truth_class_ids,
                        validation_probabilities,
                        labels=self.class_names,
                    )
                }
            )

    return ROCCurveLogger()


@app.function
def train_and_log(config, wandb_settings, train_path, validation_path):
    """Train the model and complete one W&B run."""
    if wandb.run is not None:
        wandb.finish()

    with wandb.init(
        project=wandb_settings["project"],
        entity=wandb_settings["entity"],
        config=config,
    ) as run:
        run_config = run.config
        tf.random.set_seed(run_config.random_seed)
        np.random.seed(run_config.random_seed)

        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
        )
        validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

        train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=(run_config.img_width, run_config.img_height),
            batch_size=run_config.batch_size,
            class_mode="categorical",
        )
        validation_generator = validation_datagen.flow_from_directory(
            validation_path,
            target_size=(run_config.img_width, run_config.img_height),
            batch_size=run_config.batch_size,
            class_mode="categorical",
            # Shuffle so the validation subset logged to the ROC chart samples
            # every class instead of the first few directory-ordered classes.
            shuffle=True,
            seed=run_config.random_seed,
        )

        model = build_model(run_config.fc_size, run_config.num_classes)
        callbacks = [
            WandbMetricsLogger(),
            make_roc_curve_logger(
                run,
                validation_generator,
                run_config.num_log_batches,
            ),
        ]
        history = model.fit(
            train_generator,
            steps_per_epoch=max(1, run_config.num_train // run_config.batch_size),
            epochs=run_config.pretrain_epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            validation_steps=max(1, run_config.num_val // run_config.batch_size),
        )

        final_validation_accuracy = history.history.get("val_accuracy", [None])[-1]
        result = {
            "run_url": run.url,
            "final_validation_accuracy": (
                float(final_validation_accuracy)
                if final_validation_accuracy is not None
                else None
            ),
        }

    return result


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Run the experiment

    After connecting to W&B above, submit this training form to materialize the remote dataset in the runtime cache if necessary, download the InceptionV3 weights if necessary, train the model, and create one run in the `custom_roc_curve` project. A later submission finishes any prior active run before it starts another.
    """)
    return


@app.cell(hide_code=True)
def _(wandb_settings):
    _workflow_ready = callable(ensure_dataset) and callable(train_and_log)
    _num_train_input = mo.ui.number(
        start=32,
        stop=10_000,
        step=1,
        value=100,
        label="Training examples",
    )
    _num_epochs_input = mo.ui.number(
        start=1,
        stop=20,
        step=1,
        value=1,
        label="Epochs",
    )
    training_form = (
        mo.md("{num_train}\n\n{num_epochs}")
        .batch(
            num_train=_num_train_input,
            num_epochs=_num_epochs_input,
        )
        .form(
            submit_button_label="Prepare data, train, and log to W&B",
            submit_button_disabled=not _workflow_ready,
            bordered=True,
        )
    )
    _destination = (
        f"`{wandb_settings['entity']}/{wandb_settings['project']}`"
        if wandb_settings["entity"]
        else f"`{wandb_settings['project']}` in your default entity"
    )
    mo.vstack(
        [
            mo.md(f"This run will be created in {_destination}."),
            training_form,
        ]
    )
    return (training_form,)


@app.cell(hide_code=True)
def _(training_form, wandb_settings):
    mo.stop(
        training_form.value is None,
        mo.callout(
            mo.md("Review the workflow above, then submit the form to begin."),
            kind="info",
        ),
    )

    training_request = {
        "wandb_settings": dict(wandb_settings),
        "num_train": int(training_form.value["num_train"]),
        "num_epochs": int(training_form.value["num_epochs"]),
    }
    return (training_request,)


@app.cell
def _(training_request):
    # Experiment configuration saved to W&B.
    config_defaults = {
        "num_train": training_request["num_train"],
        "num_val": 500,
        "num_classes": 10,
        "fc_size": 1024,
        "img_width": 299,
        "img_height": 299,
        "batch_size": 32,
        "pretrain_epochs": training_request["num_epochs"],
        "num_log_batches": 15,
        "random_seed": 23,
    }
    config_defaults
    return (config_defaults,)


@app.cell(hide_code=True)
def _(config_defaults, dataset_fs):
    train_path, validation_path, dataset_status = ensure_dataset(dataset_fs)
    mo.callout(
        mo.md(
            f"{dataset_status} Training will use "
            f"{config_defaults['num_train']:,} images per epoch."
        ),
        kind="success",
    )
    return train_path, validation_path


@app.cell
def _(config_defaults, train_path, training_request, validation_path):
    experiment_result = train_and_log(
        config=config_defaults,
        wandb_settings=training_request["wandb_settings"],
        train_path=train_path,
        validation_path=validation_path,
    )
    return (experiment_result,)


@app.cell(hide_code=True)
def _(experiment_result):
    _accuracy = experiment_result["final_validation_accuracy"]
    _accuracy_note = (
        f" The final validation accuracy was `{_accuracy:.3f}`."
        if _accuracy is not None
        else ""
    )
    mo.callout(
        mo.md(
            f"Training finished.{_accuracy_note} "
            f"[Open the run in W&B]({experiment_result['run_url']}) and find "
            "`roc_curve` under **Custom Charts** to inspect the ROC curves."
        ),
        kind="success",
    )
    return


if __name__ == "__main__":
    app.run()
