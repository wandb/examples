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
"""Fine-tune InceptionV3 and log a confusion matrix to W&B."""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="Plot a Confusion Matrix with W&B")

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
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.callbacks import Callback
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

    DATASET_URL = "https://storage.googleapis.com/wandb_datasets/nature_12K.zip"
    PROJECT_NAME = "confusion_matrix"
    RUNTIME_CACHE_ROOT = (
        Path(tempfile.gettempdir()) / "wandb-examples" / "inaturalist-12k"
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Plot a Confusion Matrix with W&B

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/log-a-confusion-matrix-with-w-b/log_a_confusion_matrix_with_w_b.py/server)

    [View the original Colab notebook](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log_a_Confusion_Matrix_with_W&B.ipynb).

    <!--- @wandbcode{confusion_matrix} -->

    How to log a [confusion matrix](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html) with [Vega](https://vega.github.io/vega/docs/) in [Weights & Biases](https://www.wandb.com).

    ## Method: `wandb.plot.confusion_matrix()`

    - More info and customization details: [Confusion Matrix](https://wandb.ai/wandb/plots/reports/Confusion-Matrix--VmlldzozMDg1NTM)
    - More examples in this W&B project: [Custom Charts](https://app.wandb.ai/demo-team/custom-charts).

    This notebook explores a transfer learning problem: finetuning InceptionV3 with ImageNet weights to identify 10 types of living things (birds, plants, insects, etc) from 10K photos via [iNaturalist 2017](https://github.com/visipedia/inat_comp).

    ![confusion_matrix](https://i.imgur.com/rvKx8RF.png)

    Note: Hyperparameters like number of epochs and training dataset size are set to minimum values here for demo efficiency. On the full training data, the model should get to the low 80s in validation accuracy within an epoch or so.
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

    The iNaturalist sample contains 10,000 training images and 2,000 validation images, evenly distributed across 10 classes of living things like birds, insects, plants, and mammals. **The archive is approximately 3.6 GB, so the first transfer and extraction can take several minutes.**

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

    Feel free to try different values for "NUM_TRAIN" and "NUM_EPOCHS" below so you can see a variety of confusion matrices (generally better ones with more training examples/longer training time).
    """)
    return


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


@app.function
def pretrain(config_defaults, wandb_settings, train_data, val_data):
    """Main training loop.

    This is called "pretrain" because it freezes the InceptionV3 layers of the
    model and only trains the new top layers on the new data. A subsequent
    training phase would unfreeze all the layers and finetune the whole model
    on the new data.
    """
    if wandb.run is not None:
        wandb.finish()

    checkpoint_cache = RUNTIME_CACHE_ROOT / "checkpoints"
    checkpoint_cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="confusion-matrix-", dir=checkpoint_cache
    ) as checkpoint_directory:
        checkpoint_path = Path(checkpoint_directory) / "checkpoint.keras"
        # The checkpoint directory remains alive until the run context finishes
        # and W&B has flushed the checkpoint Artifact.
        with wandb.init(
            project=wandb_settings["project"],
            entity=wandb_settings["entity"],
            config=config_defaults,
        ) as run:
            cfg = run.config

            # Set the random seed.
            tf.random.set_seed(cfg.random_seed)

            # Also set the NumPy seed to control the train/validation data.
            np.random.seed(cfg.random_seed)

            # Create train and validation data generators.
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
            )
            validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

            train_generator = train_datagen.flow_from_directory(
                train_data,
                target_size=(cfg.img_width, cfg.img_height),
                batch_size=cfg.batch_size,
                class_mode="categorical",
            )
            val_generator = validation_datagen.flow_from_directory(
                val_data,
                target_size=(cfg.img_width, cfg.img_height),
                batch_size=cfg.batch_size,
                class_mode="categorical",
            )

            # Instantiate the model and callbacks.
            model = build_model(cfg.fc_size, cfg.num_classes)
            callbacks = [
                WandbMetricsLogger(),
                WandbModelCheckpoint(str(checkpoint_path)),
                PRMetrics(
                    run,
                    val_generator,
                    cfg.num_log_batches,
                ),
            ]

            # Train!
            history = model.fit(
                train_generator,
                steps_per_epoch=max(1, cfg.num_train // cfg.batch_size),
                epochs=cfg.pretrain_epochs,
                validation_data=val_generator,
                callbacks=callbacks,
                validation_steps=max(1, cfg.num_val // cfg.batch_size),
            )

            final_validation_accuracy = history.history.get(
                "val_accuracy", [None]
            )[-1]
            result = {
                "run_url": run.url,
                "final_validation_accuracy": (
                    float(final_validation_accuracy)
                    if final_validation_accuracy is not None
                    else None
                ),
            }

    return result


@app.class_definition
class PRMetrics(Callback):
    """Custom callback to compute metrics at the end of each training epoch."""

    def __init__(self, run, generator=None, num_log_batches=1):
        super().__init__()
        self.run = run
        self.generator = generator
        self.num_batches = num_log_batches

        # Store full names of classes in class-ID order.
        self.flat_class_names = [
            name
            for name, _class_id in sorted(
                generator.class_indices.items(), key=lambda item: item[1]
            )
        ]

    def on_epoch_end(self, epoch, logs=None):
        # Collect validation data and ground truth labels from the generator.
        val_data, val_labels = zip(
            *(self.generator[index] for index in range(self.num_batches))
        )
        val_data, val_labels = np.vstack(val_data), np.vstack(val_labels)

        # Use the trained model to generate predictions for the given number
        # of validation data batches (num_batches).
        val_predictions = self.model.predict(val_data, verbose=0)
        ground_truth_class_ids = val_labels.argmax(axis=1)

        # Take the argmax for each set of prediction scores to return the class
        # ID of the highest-confidence prediction.
        top_pred_ids = val_predictions.argmax(axis=1)

        # Log the confusion matrix. The key "conf_mat" is the ID of the plot;
        # keep it stable so subsequent runs appear on the same plot.
        self.run.log(
            {
                "conf_mat": wandb.plot.confusion_matrix(
                    probs=None,
                    preds=top_pred_ids,
                    y_true=ground_truth_class_ids,
                    class_names=self.flat_class_names,
                )
            }
        )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Run the experiment

    After connecting to W&B above, submit this training form to materialize the remote dataset in the runtime cache if necessary, download the InceptionV3 weights if necessary, train the model, and create one run in the `confusion_matrix` project. A later submission finishes any prior active run before it starts another.
    """)
    return


@app.cell(hide_code=True)
def _(wandb_settings):
    _workflow_ready = callable(ensure_dataset) and callable(pretrain)
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
    # Experiment configuration saved to W&B
    config_defaults = {
        # Number of images used to train; set low for demo training speed.
        # You can set this up to 10,000 for the full dataset.
        # GOOD CONFIG TO TRY: 100, 500, 1000, 2000
        "num_train": training_request["num_train"],
        # Number of images used to validate; set low for demo training speed.
        # You can set this up to 2,000 for the full dataset.
        "num_val": 500,
        "num_classes": 10,
        "fc_size": 1024,

        # InceptionV3 settings
        "img_width": 299,
        "img_height": 299,
        "batch_size": 32,

        # Number of epochs; set low for demo training speed.
        # GOOD CONFIG TO TRY: 3, 5, 10
        "pretrain_epochs": training_request["num_epochs"],
        # Number of validation data batches used to compute metrics at the end
        # of each epoch.
        "num_log_batches": 15,
        # Random seed
        "random_seed": 23,
    }
    config_defaults
    return (config_defaults,)


@app.cell(hide_code=True)
def _(config_defaults, dataset_fs):
    train_data, val_data, dataset_status = ensure_dataset(dataset_fs)
    mo.callout(
        mo.md(
            f"{dataset_status} Training will use "
            f"{config_defaults['num_train']:,} images per epoch."
        ),
        kind="success",
    )
    return train_data, val_data


@app.cell
def _(config_defaults, train_data, training_request, val_data):
    # Run this cell to launch your experiment! Charts appear in the run page
    # under "Media" or "Custom Charts", which you may need to expand.
    experiment_result = pretrain(
        config_defaults=config_defaults,
        wandb_settings=training_request["wandb_settings"],
        train_data=train_data,
        val_data=val_data,
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
            "`conf_mat` under **Custom Charts** to inspect the confusion matrix."
        ),
        kind="success",
    )
    return


if __name__ == "__main__":
    app.run()
