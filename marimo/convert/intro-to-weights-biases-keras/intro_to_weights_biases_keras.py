# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "marimo>=0.24.2",
#     "tensorflow>=2.16,<3",
#     "wandb>=0.18,<1",
# ]
#
# [tool.marimo.runtime]
# on_cell_change = "lazy"
# ///
"""Learn the Weights & Biases experiment-tracking workflow with simulated and Keras runs."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    app_title="Introduction to Weights & Biases with Keras",
)

with app.setup:
    from pathlib import Path
    import random
    import tempfile

    import marimo as mo
    import tensorflow as tf

    import wandb
    from wandb import AlertLevel
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

    # Introduction to Weights & Biases with Keras

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/intro-to-weights-biases-keras/intro_to_weights_biases_keras.py/server)

    Use [Weights & Biases](https://wandb.ai/site?utm_source=keras_intro_colab&utm_medium=code&utm_campaign=keras_intro)
    for experiment tracking, model checkpointing, and collaboration. This
    tutorial starts with simulated metrics, then trains a small Keras classifier,
    and finishes with a scriptable alert.

    Running the code cells creates Weights & Biases runs, downloads MNIST, and uploads metrics
    and model checkpoints. Edit the Python configuration before running the
    examples.

    ## A shared dashboard for your experiments

    With few lines of code, you can create rich, interactive, shareable
    dashboards like this [example Weights & Biases project](https://wandb.ai/wandb/wandb_example).

    ![Example Weights & Biases dashboard](https://i.imgur.com/Pell4Oo.png)

    ## Authenticate to Weights & Biases

    Set `entity` to your Weights & Biases team name, or leave it as `None` to use your
    default entity. `wandb.login()` uses
    your saved credentials or prompts for an [API key](https://wandb.ai/authorize).
    In a locally hosted notebook, you can also set the `WANDB_API_KEY` secret.
    """)
    return


@app.cell
def _():
    entity = None
    simulation_project = "basic-intro"
    training_project = "keras-intro"

    wandb.login()
    return entity, simulation_project, training_project


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Run simulated experiments

    The basic Weights & Biases workflow has three steps:

    1. Start a run with `wandb.init()` and record its configuration.
    2. Log metrics from training.
    3. Open the printed run links to compare results in the dashboard.

    The next cell creates five experiments by default. These accuracy and loss
    curves are synthetic; no model is trained. A fixed seed makes the examples
    reproducible. Each context manager finishes its run before the next starts.
    """)
    return


@app.cell
def _(entity, simulation_project):
    simulation_count = 5
    simulation_seed = 42
    simulation_rng = random.Random(simulation_seed)

    for simulation_index in range(simulation_count):
        with wandb.init(
            entity=entity,
            project=simulation_project,
            name=f"keras-intro-simulation-{simulation_index + 1}",
            config={"dataset": "synthetic", "seed": simulation_seed, "epochs": 8},
        ) as simulation_run:
            offset = simulation_rng.random() / 5
            for simulation_epoch in range(2, 10):
                simulated_accuracy = 1 - 2**(-simulation_epoch) - simulation_rng.random() / simulation_epoch - offset
                simulated_loss = 2**(-simulation_epoch) + simulation_rng.random() / simulation_epoch + offset
                simulation_run.log({
                    "epoch": simulation_epoch - 1,
                    "accuracy": simulated_accuracy,
                    "loss": simulated_loss,
                })
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Train a Keras classifier

    This example downloads MNIST, trains a small dense classifier, and streams
    metrics to Weights & Biases. Two callbacks handle the logging:

    - `WandbMetricsLogger` records Keras training and validation metrics.
    - `WandbModelCheckpoint` saves each epoch's model and logs it as a Weights & Biases Artifact.

    Checkpoints are written to a temporary directory and removed after Weights & Biases has
    finished the run, so rerunning this notebook does not leave model files in
    the repository.

    ### Configure the experiments

    The dictionary below defines the model's layers, optimizer, and training
    settings. We train five fresh models with reproducibly sampled dropout
    rates. Weights & Biases records each model's settings so you can compare
    dropout against validation loss and accuracy. Reduce `experiment_count`
    or `epochs` for a quicker run.
    """)
    return


@app.cell
def _():
    training_config = {
        "layer_1": 512,
        "activation_1": "relu",
        "layer_2": 10,
        "activation_2": "softmax",
        "optimizer": "sgd",
        "loss": "sparse_categorical_crossentropy",
        "metric": "accuracy",
        "epochs": 6,
        "batch_size": 256,
        "seed": 42,
        "dataset": "MNIST",
    }
    experiment_count = 5
    dropout_rng = random.Random(training_config["seed"])
    dropout_values = [dropout_rng.uniform(0.01, 0.80) for _ in range(experiment_count)]
    return dropout_values, training_config


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Prepare MNIST once

    The next cell downloads MNIST if needed, normalizes pixels to `[0, 1]`,
    and samples 12,000 training images and 500 held-out images for a shorter demo.
    All experiments reuse these arrays. We use the held-out images for validation
    during training; they are not an independent final test.
    """)
    return


@app.cell
def _():
    (x_train, y_train), (x_validation, y_validation) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[::5].astype("float32") / 255.0
    y_train = y_train[::5]
    x_validation = x_validation[::20].astype("float32") / 255.0
    y_validation = y_validation[::20]
    return x_train, x_validation, y_train, y_validation


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Train and log with Keras callbacks

    Each `wandb.init()` creates a separate `training_run`. A fresh model and
    optimizer are built for each dropout value. The seed is reset for each
    experiment to make comparisons easier to reproduce.

    `WandbMetricsLogger` sends training and validation metrics after each epoch.
    `WandbModelCheckpoint` saves the model after each epoch and uploads the
    checkpoint as a model Artifact. The `{epoch:02d}` filename placeholder keeps
    checkpoints separate. The run's summary also records its final validation accuracy.
    """)
    return


@app.cell
def _(
    dropout_values,
    entity,
    training_config,
    training_project,
    x_train,
    x_validation,
    y_train,
    y_validation,
):
    for experiment_index, dropout in enumerate(dropout_values):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(training_config["seed"])
        with tempfile.TemporaryDirectory(prefix="wandb-keras-checkpoints-") as checkpoint_directory:
            with wandb.init(
                entity=entity,
                project=training_project,
                name=f"keras-mnist-{experiment_index + 1}-dropout-{dropout:.2f}",
                config={**training_config, "dropout": dropout},
            ) as training_run:
                run_config = training_run.config
                model = tf.keras.Sequential([
                    tf.keras.Input(shape=(28, 28)),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(run_config.layer_1, activation=run_config.activation_1),
                    tf.keras.layers.Dropout(run_config.dropout),
                    tf.keras.layers.Dense(run_config.layer_2, activation=run_config.activation_2),
                ])
                model.compile(
                    optimizer=run_config.optimizer,
                    loss=run_config.loss,
                    metrics=[run_config.metric],
                )
                checkpoint_path = str(Path(checkpoint_directory) / "model-{epoch:02d}.keras")
                history = model.fit(
                    x=x_train,
                    y=y_train,
                    epochs=run_config.epochs,
                    batch_size=run_config.batch_size,
                    validation_data=(x_validation, y_validation),
                    callbacks=[
                        WandbMetricsLogger(log_freq="epoch"),
                        WandbModelCheckpoint(filepath=checkpoint_path),
                    ],
                    verbose=2,
                )
                training_run.summary["final_validation_accuracy"] = float(history.history["val_accuracy"][-1])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Inspect your results

    Open the project link printed by Weights & Biases. Compare dropout values in the run
    configurations against the `epoch/val_loss` and `epoch/val_accuracy` charts
    and the `final_validation_accuracy` summary. In a run's **Artifacts** tab,
    find the saved Keras model. You can download it and load it with
    `tf.keras.models.load_model()` to reuse the checkpoint.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Try Weights & Biases Alerts

    [Weights & Biases Alerts](https://docs.wandb.ai/models/runs/alert) can notify you through
    Slack or email when code detects a condition such as low accuracy. Before
    trying this section, enable **Scriptable run alerts** and configure a
    destination in your Weights & Biases user settings.

    The next cell logs synthetic accuracy values between zero and one. The
    first value at or below `accuracy_threshold` triggers `alert_run.alert()`
    and ends the loop. These values are independent of the MNIST results.

    Set the threshold and random seed below, then run the cell to create an
    alert-demo run. With notifications configured, this can send a real Slack
    or email alert. `wait_duration=300` limits repeated alerts with the same
    title to one every five minutes.
    """)
    return


@app.cell
def _(entity, training_project):
    accuracy_threshold = 0.3
    max_alert_steps = 1000
    alert_seed = 7

    alert_rng = random.Random(alert_seed)
    with wandb.init(
        entity=entity,
        project=training_project,
        name="keras-low-accuracy-alert-demo",
        job_type="alert-demo",
        config={"accuracy_threshold": accuracy_threshold, "seed": alert_seed},
    ) as alert_run:
        for alert_step in range(max_alert_steps):
            alert_accuracy = round(alert_rng.random(), 3)
            alert_run.log({"step": alert_step, "accuracy": alert_accuracy})
            if alert_accuracy <= accuracy_threshold:
                alert_run.alert(
                    title="Low accuracy",
                    text=(
                        f"Accuracy {alert_accuracy} at step {alert_step} is at or below "
                        f"the threshold {accuracy_threshold}."
                    ),
                    level=AlertLevel.WARN,
                    wait_duration=300,
                )
                print(f"Alert triggered: {alert_run.url}")
                break
        else:
            print("No value crossed the threshold; no alert was sent.")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Verify and next steps

    In your W&B workspace, check:

    - **`basic-intro`**: each simulation has eight logged accuracy and loss values.
    - **`keras-intro`**: compare `dropout` in each training run's config with
      `epoch/val_loss`, `epoch/val_accuracy`, and `final_validation_accuracy`.
      Open a training run's **Artifacts** tab to find its per-epoch Keras models.
    - **Alert demo**: inspect the accuracy history in the run named
      `keras-low-accuracy-alert-demo`. With the default seed, the second value
      crosses the threshold. Check your configured notification destination
      for the warning.

    Continue with
    [Organizing hyperparameter sweeps in PyTorch](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/organizing-hyperparameter-sweeps-in-pytorch-with-w-b/organizing_hyperparameter_sweeps_in_pytorch_with_w_b.py/server)
    to automate a collection of training runs.
    """)
    return


if __name__ == "__main__":
    app.run()
