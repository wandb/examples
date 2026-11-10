# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "marimo>=0.24.2",
#     "numpy>=1.26,<3",
#     "tensorflow>=2.16,<3",
#     "wandb>=0.18,<1",
# ]
#
# [tool.marimo.runtime]
# on_cell_change = "lazy"
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="TensorFlow hyperparameter sweeps with Weights & Biases",
)

with app.setup:
    import marimo as mo
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras

    import wandb
    from wandb.integration.keras import WandbMetricsLogger


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

    # Tune a TensorFlow model with Weights & Biases Sweeps

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/hyperparameter-optimization-in-tensorflow-using-w-b-sweeps/hyperparameter_optimization_in_tensorflow_using_w_b_sweeps.py/server)

    A sweep is a collection of experiments that explores a hyperparameter search
    space. In this notebook, each trial trains the same MNIST digit classifier with a
    different batch size and learning rate. Weights & Biases selects the settings, and its
    Keras integration logs the metrics used to compare trials.

    First define a search space and a metric to optimize. Then write a function
    that trains one model using `run.config`. Finally, start a sweep agent to
    request settings and call that function for each trial.
    This example launches up to ten trials with two epochs each.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Authenticate with Weights & Biases

    Paste your [Forge API key](https://docs.coreweave.com/products/wandb/platform/hosting/iam/api-keys) into the password field below. Keep your key out of the notebook’s source code.
    """)
    return


@app.cell(hide_code=True)
def _():
    def _notify_submission(_value):
        mo.status.toast(
            "Credentials submitted",
            "Run the next cell to verify them with Weights & Biases.",
        )

    wandb_api_form = (
        mo.md("{api_key}")
        .batch(
            api_key=mo.ui.text(
                kind="password",
                label="Forge API key",
                full_width=True,
            )
        )
        .form(
            submit_button_label="Connect to Weights & Biases",
            bordered=True,
            on_change=_notify_submission,
        )
    )

    wandb_api_form
    return (wandb_api_form,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    After submitting the API key, run the following cell to verify the
    credentials and load your available Weights & Biases entities.
    """)
    return


@app.cell(hide_code=True)
def _(wandb_api_form):
    from wandb.sdk.lib.service.service_connection import (
        WandbApiFailedError as _WandbApiFailedError,
    )

    mo.stop(wandb_api_form.value is None)

    _submission = wandb_api_form.value
    _api_key = _submission["api_key"].strip()

    try:
        _login_ok = wandb.login(
            key=_api_key or None,
            relogin=bool(_api_key),
            verify=True,
            prompt=False,
        )
    except (wandb.errors.Error, ValueError):
        _login_ok = False

    mo.stop(
        not _login_ok,
        mo.callout("Check your API key or saved credentials, then connect again.", kind="danger"),
    )

    try:
        _api = wandb.Api(timeout=15)
        _viewer = _api.viewer
    except (wandb.errors.Error, _WandbApiFailedError, ValueError):
        mo.stop(
            True,
            mo.callout(
                "Weights & Biases verified the credentials but could not retrieve your account entities.",
                kind="danger",
            ),
        )

    available_entities = sorted({
        _value
        for _value in (
            getattr(_viewer, "username", None),
            getattr(_viewer, "entity", None),
            *_viewer.teams,
        )
        if _value
    })
    mo.stop(
        not available_entities,
        mo.callout(
            "Weights & Biases authenticated successfully, but no personal or team entities were found.",
            kind="danger",
        ),
    )

    mo.callout(
        mo.md("✓ Credentials verified. Specify your team entity below."),
        kind="success",
    )
    return (available_entities,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Set your team and project

    Run the following cell and specify your:

    - Weights & Biases team entity.
    - A name for your [project](https://docs.coreweave.com/products/wandb/track/project-page#projects). The default project is `sweeps-tensorflow`.

    Then select **Use this Weights & Biases target**.

    > If the project doesn't exist, Weights & Biases creates it when the first run starts, provided you have write access.
    """)
    return


@app.cell(hide_code=True)
def _(available_entities):
    def _validate_wandb_target(value):
        if value is None:
            return None
        if not value["entity"]:
            return "Select a Weights & Biases entity."
        if not value["project"].strip():
            return "Enter a project name."
        return None

    def _notify_wandb_submission(_value):
        mo.status.toast(
            "Weights & Biases target selected",
            "Run the next cell to confirm the destination.",
        )

    wandb_setup_form = (
        mo.md("{entity}\n\n{project}")
        .batch(
            entity=mo.ui.dropdown(
                options=available_entities,
                value=available_entities[0],
                label="Weights & Biases entity",
                searchable=True,
                full_width=True,
            ),
            project=mo.ui.text(
                value="sweeps-tensorflow",
                label="Project",
                full_width=True,
            ),
        )
        .form(
            submit_button_label="Use this Weights & Biases target",
            validate=_validate_wandb_target,
            on_change=_notify_wandb_submission,
            bordered=True,
        )
    )

    wandb_setup_form
    return (wandb_setup_form,)




@app.cell(hide_code=True)
def _(wandb_setup_form):
    mo.stop(wandb_setup_form.value is None)

    _submission = wandb_setup_form.value
    _entity = _submission["entity"]
    _project = _submission["project"].strip()

    entity = _entity
    project = _project

    return entity, project


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Prepare data shared by every trial

    MNIST contains 28 × 28 grayscale images with integer labels from 0 to 9.
    Flatten each image to 784 features and convert pixel values from 0–255 to
    floating-point values from 0–1.

    Load and normalize the data once. Use the first 50,000 training examples
    for fitting and the remaining 10,000 for validation, keeping that split
    fixed across trials. The official test set is reserved for evaluating the
    model after hyperparameter selection.
    """)
    return


@app.cell
def _():
    (images, labels), _ = keras.datasets.mnist.load_data()
    images = images.reshape(-1, 784).astype(np.float32) / 255.0

    train_images, train_labels = images[:-10000], labels[:-10000]
    val_images, val_labels = images[-10000:], labels[-10000:]
    return train_images, train_labels, val_images, val_labels


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Keep the model architecture fixed

    Each trial creates a fresh multilayer perceptron with two hidden layers of
    64 ReLU units. The ten outputs are class scores, or logits. The training
    function will use a loss with `from_logits=True`, so the final layer does
    not apply softmax.

    Keeping the architecture fixed lets us compare the effect of the two
    hyperparameters in the search space.
    """)
    return


@app.function
def build_model():
    inputs = keras.Input(shape=(784,), name="digits")
    hidden = keras.layers.Dense(64, activation="relu")(inputs)
    hidden = keras.layers.Dense(64, activation="relu")(hidden)
    outputs = keras.layers.Dense(10, name="predictions")(hidden)
    return keras.Model(inputs=inputs, outputs=outputs)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Define the search space and objective

    `method="random"` samples a batch size and learning rate from the listed
    values for each trial. The objective is to minimize `epoch/val_loss`.
    That exact key is logged by `WandbMetricsLogger` at the end of each epoch.
    The sweep metric name must match the logged name exactly.

    Hyperband can stop weaker trials early. Its `min_iter=1` sets the first
    comparison point after one report of the objective metric, which here
    corresponds to one epoch. Checks happen asynchronously, so these short
    two-epoch trials may finish before they can be stopped. Longer trials
    provide more opportunity for early termination.
    """)
    return


@app.cell
def _():
    sweep_config = {
        "method": "random",
        "metric": {"name": "epoch/val_loss", "goal": "minimize"},
        "early_terminate": {"type": "hyperband", "min_iter": 1},
        "parameters": {
            "batch_size": {"values": [32, 64, 128, 256]},
            "learning_rate": {"values": [0.01, 0.005, 0.001, 0.0005, 0.0001]},
        },
    }
    return (sweep_config,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Define one complete trial

    The agent calls a function with no arguments. Inside `sweep_train`,
    `wandb.init()` starts the run with default settings. The sweep’s selected
    batch size and learning rate override those defaults in
    `run.config`. The context manager finishes the run when training ends.

    Each trial resets Keras and uses the same random seed before building a
    fresh model. The selected batch size controls the `tf.data` pipelines, and
    the selected learning rate configures SGD. The training dataset shuffles
    before batching. `model.fit(shuffle=False)` leaves that order to the
    pipeline. Validation data uses the same examples and order for every trial.

    `WandbMetricsLogger(log_freq=20)` logs batch metrics every 20 batches and
    training and validation metrics after each epoch. In addition to the
    optimized `epoch/val_loss`, inspect `epoch/acc` and `epoch/val_acc` for digit
    classification accuracy. To change the training budget, edit `epochs` in
    the defaults below.
    """)
    return


@app.cell
def _(entity, project, train_images, train_labels, val_images, val_labels):
    def sweep_train():
        config_defaults = {
            "batch_size": 64,
            "learning_rate": 0.01,
            "epochs": 2,
            "log_step": 20,
            "seed": 42,
            "architecture_name": "MLP",
            "dataset_name": "MNIST",
        }

        with wandb.init(entity=entity, project=project, config=config_defaults) as run:
            config = run.config
            keras.backend.clear_session()
            keras.utils.set_random_seed(config.seed)

            train_dataset = (
                tf.data.Dataset.from_tensor_slices((train_images, train_labels))
                .shuffle(buffer_size=1024, seed=config.seed)
                .batch(config.batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )
            val_dataset = (
                tf.data.Dataset.from_tensor_slices((val_images, val_labels))
                .batch(config.batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )

            model = build_model()
            model.compile(
                optimizer=keras.optimizers.SGD(learning_rate=config.learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
            )
            return model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=config.epochs,
                callbacks=[WandbMetricsLogger(log_freq=config.log_step)],
                shuffle=False,
                verbose=2,
            )

    return (sweep_train,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Register the sweep

    `wandb.sweep()` sends the search configuration to Weights & Biases and returns the
    sweep’s ID. Registration creates the sweep in your chosen team and project.
    Training begins when an agent starts. Weights & Biases prints a link to the sweep
    dashboard. Rerunning this cell creates a new sweep.
    """)
    return


@app.cell
def _(entity, project, sweep_config):
    sweep_id = wandb.sweep(sweep_config, entity=entity, project=project)
    return (sweep_id,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Start an agent

    The agent requests hyperparameters from Weights & Biases and calls `sweep_train` for
    each trial. `count=10` limits how many trials this agent starts. Lower the count for a quicker experiment.

    To add trials to an existing sweep, use its ID as `sweep_id` and run the
    agent cell again, keeping the same entity and project. Sweep coordination
    requires a connection to your Weights & Biases server.
    """)
    return


@app.cell
def _(entity, project, sweep_id, sweep_train):
    wandb.agent(
        sweep_id,
        function=sweep_train,
        count=10,
        entity=entity,
        project=project,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Verify the sweep results

    Open the URL printed by `wandb.sweep()`. The sweep URL ends with:

    ```
    /sweeps/<sweep-id>
    ```

    Navigate to the **Sweep** panel to view the

    - Parallel coordinates
    - paramater importance
    - scatter plot of runs

    Compare `epoch/val_loss` and `epoch/val_acc`, then use parallel coordinates to connect parameter choices to validation loss. Parameter importance becomes more useful as the sweep gains trials, so do not over-interpret a small sample.

    See the [Sweeps visualization guide](https://docs.coreweave.com/products/wandb/sweeps/visualize-sweep-results)
    for these views.

    ## Next steps

    Change the method to Bayesian search or narrow the learning-rate range, then
    register a new sweep.

    After choosing a configuration, retrain it and return to your Weights & Biases projects to compare results.
    """)
    return


if __name__ == "__main__":
    app.run()
