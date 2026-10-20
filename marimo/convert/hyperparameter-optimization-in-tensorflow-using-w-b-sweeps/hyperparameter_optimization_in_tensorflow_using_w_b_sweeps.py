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

__generated_with = "0.25.1"
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
    space. Here, each trial trains the same MNIST digit classifier with a
    different batch size and learning rate. Weights & Biases selects the settings, and its
    Keras integration logs the metrics used to compare trials.

    The workflow has three parts: define the search space, write a function that
    trains one model using `run.config`, and start an agent to run the trials.
    This example launches up to ten trials with two epochs each.

    ## Choose your Weights & Biases project and log in

    Set `entity` to your username or team, or leave it as `None` to use your
    default entity. The same `project` is used for the sweep, agent, and runs.
    `wandb.login()` uses existing credentials or prompts for an
    [API key](https://wandb.ai/authorize). In a hosted notebook, you can also
    supply the key through the `WANDB_API_KEY` secret.
    """)
    return


@app.cell
def _():
    entity = None
    project = "sweeps-tensorflow"

    wandb.login()
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
    That exact key is logged by `WandbMetricsLogger` at the end of each epoch;
    the sweep metric name must match the logged name.

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
    `wandb.init()` starts the run and supplies default settings; the sweep's
    selected batch size and learning rate override those defaults in
    `run.config`. The context manager finishes the run when training ends.

    Each trial resets Keras and uses the same random seed before building a
    fresh model. The selected batch size controls the `tf.data` pipelines, and
    the selected learning rate configures SGD. The training dataset shuffles
    before batching; `model.fit(shuffle=False)` leaves that order to the
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
    sweep's ID. This creates the sweep in your chosen entity and project;
    training begins when an agent starts. Weights & Biases prints a link to the sweep
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
    each trial. `count=10` limits how many trials this agent starts; lower it
    for a quicker experiment. The call waits while trials run.

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
    ## Verify and compare results

    Open the sweep link printed during registration. In the runs table, check
    that `batch_size` and `learning_rate` vary between trials. Each run's config
    also records the fixed epoch budget, seed, architecture, and dataset.

    Compare `epoch/val_loss` across runs and inspect the corresponding
    `epoch/val_acc` curves. Use the parallel-coordinates plot to connect
    hyperparameter choices to validation loss. Parameter importance becomes
    more informative as you collect more trials. This reference dashboard
    illustrates the views to look for:

    ![Example Weights & Biases Sweeps dashboard](https://i.imgur.com/AN0qnpC.png)

    Try `method="bayes"` or expand the search space, then register a new sweep
    and compare the results. After selecting hyperparameters, evaluate the
    chosen model on the reserved test set to estimate performance on unseen
    data.
    """)
    return


if __name__ == "__main__":
    app.run()
