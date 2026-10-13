# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "marimo>=0.24.2",
#     "numpy>=1.26,<3",
#     "tensorflow>=2.16,<3",
#     "wandb>=0.18,<1",
# ]
# ///
"""Optimize a TensorFlow classifier with W&B Sweeps."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    app_title="TensorFlow hyperparameter sweeps with W&B",
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
    # Weights & Biases Sweep + TensorFlow 2.x

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/hyperparameter-optimization-in-tensorflow-using-w-b-sweeps/hyperparameter_optimization_in_tensorflow_using_w_b_sweeps.py/server)

    <img src="https://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    This notebook shows the smallest practical loop for tuning TensorFlow
    hyperparameters with Weights & Biases Sweeps: define a search space, turn one
    training run into a reusable function, launch an agent, and inspect the
    results in W&B.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A sweep is a coordinated set of experiment runs. W&B chooses hyperparameter
    values, starts runs through your training function, logs metrics from each
    trial, and builds dashboards for comparing the results.

    By the end of the notebook, your sweep dashboard will look conceptually like
    this example:

    ![Example W&B Sweeps dashboard](https://i.imgur.com/AN0qnpC.png)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    [Weights & Biases Sweeps](https://docs.coreweave.com/products/wandb/sweeps)
    automate hyperparameter search and visualize rich, interactive experiment
    tracking. They support search methods such as random search, grid search, and
    Bayesian optimization, and they can scale from this single-notebook example to
    agents running on many machines.

    ## What this notebook covers

    The cells below walk through the core W&B Sweeps pattern:

    * Authenticate with W&B.
    * Prepare a small TensorFlow image-classification problem.
    * Log training and validation metrics from Keras.
    * Define a sweep configuration for batch size and learning rate.
    * Launch a W&B agent that runs multiple trials and reports results.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 1: Import W&B and log in

    The next two cells collect your W&B credentials and call `wandb.login()`.
    W&B needs an authenticated session before it can create a sweep or sync
    training metrics.

    Run the following cell and provide your [Forge API key](https://wandb.ai/authorize) and the name of your team. Then click **Connect to Weights & Biases**.
    """)
    return


@app.cell(hide_code=True)
def _():
    wandb_login_form = (
        mo.md("{api_key}\n\n{entity}")
        .batch(
            api_key=mo.ui.text(
                kind="password",
                label="Forge API key (optional)",
                placeholder="Paste a key or use configured credentials",
                full_width=True,
            ),
            entity=mo.ui.text(
                label="Forge team name (optional)",
                placeholder="Leave blank to use your default entity",
                full_width=True,
            ),
        )
        .form(submit_button_label="Connect to Weights & Biases", bordered=True)
    )
    wandb_login_form
    return (wandb_login_form,)


@app.cell(hide_code=True)
def _(wandb_login_form):
    mo.stop(
        wandb_login_form.value is None,
        mo.callout(
            mo.md("Submit the form above to authenticate before creating a sweep."),
            kind="info",
        ),
    )

    _submitted = wandb_login_form.value
    _api_key = _submitted["api_key"].strip()
    try:
        _login_ok = wandb.login(key=_api_key or None, relogin=bool(_api_key))
    except (wandb.errors.Error, ValueError):
        _login_ok = False

    mo.stop(
        not _login_ok,
        mo.callout(
            mo.md(
                "W&B authentication did not complete. Check the API key or "
                "molab Secrets configuration and submit again."
            ),
            kind="danger",
        ),
    )

    mo.callout(mo.md("Connected to W&B."), kind="success")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 2: Prepare the Dataset

    The following function loads MNIST, normalizes pixel values, flattens each
    image into a 784-feature vector, and returns `tf.data.Dataset` objects for
    training and validation.

    The function accepts `batch_size` because the sweep will tune that value. Each
    W&B trial can therefore build datasets that match the hyperparameters chosen
    for that run.
    """)
    return


@app.function
def prepare_datasets(batch_size):
    # Prepare the training dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = np.reshape(x_train, (-1, 784))
    x_test = np.reshape(x_test, (-1, 784))

    # build input pipeline using tf.data
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = (
        train_dataset.shuffle(buffer_size=1024)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return train_dataset, val_dataset


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 3: Define the Model and Training Workflow

    The next function builds the TensorFlow model that every sweep trial will
    train. We keep the architecture fixed so the sweep can focus on two
    hyperparameters: batch size and learning rate.
    """)
    return


@app.function
def Model():
    inputs = keras.Input(shape=(784,), name="digits")
    x1 = keras.layers.Dense(64, activation="relu")(inputs)
    x2 = keras.layers.Dense(64, activation="relu")(x1)
    outputs = keras.layers.Dense(10, name="predictions")(x2)

    return keras.Model(inputs=inputs, outputs=outputs)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 4: Log metrics with `WandbMetricsLogger`

    The next function wraps `model.fit()` and attaches W&B's Keras metrics
    logger. `WandbMetricsLogger` receives the training and validation metrics
    produced by Keras and sends them to the active W&B run.

    With `log_freq=20`, it logs batch metrics every 20 batches and epoch metrics
    at the end of each epoch. Those logged values are what the sweep dashboard
    uses to compare trials.
    """)
    return


@app.function
def train(
    train_dataset,
    val_dataset,
    model,
    epochs=10,
    log_step=200,
):
    # 3️⃣ log metrics using WandbMetricsLogger
    return model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[WandbMetricsLogger(log_freq=log_step)],
        verbose=2,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 5: Configure the Sweep

    The next cell defines the sweep configuration dictionary. This is the contract
    between your training code and W&B: it tells W&B which hyperparameters to try,
    how to sample them, which metric to optimize, and when to stop weaker runs
    early.

    * Define the hyperparameters you're sweeping over.
    * Choose a hyperparameter search method. W&B supports `random`, `grid`, and `bayes`.
    * Provide an objective metric; here we minimize `epoch/val_loss`, the validation-loss key logged by `WandbMetricsLogger`.
    * Use `hyperband` for early termination of poorly-performing runs.
    """)
    return


@app.cell
def _():
    sweep_config = {
        "method": "random",
        "metric": {
            "name": "epoch/val_loss",
            "goal": "minimize",
        },
        "early_terminate": {
            "type": "hyperband",
            # Trials run for two epochs, so pruning must be allowed after epoch one.
            "min_iter": 1,
        },
        "parameters": {
            "batch_size": {
                "values": [32, 64, 128, 256],
            },
            "learning_rate": {
                "values": [0.01, 0.005, 0.001, 0.0005, 0.0001],
            },
        },
    }
    return (sweep_config,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 6: Wrap the Training Workflow

    A W&B agent needs a zero-argument training function. The next function,
    `sweep_train`, is one complete trial: it starts a W&B run, reads the
    hyperparameters selected for that run from `run.config`, prepares data,
    builds and compiles the model, trains it, and lets W&B collect metrics.

    The defaults passed to `wandb.init()` make the function runnable on its own,
    while sweep-selected values override `batch_size` and `learning_rate` during
    agent runs.
    """)
    return


@app.function
def sweep_train():
    # Set default values
    config_defaults = {
        "batch_size": 64,
        "learning_rate": 0.01,
        # Specify the other hyperparameters to the configuration, if any
        "epochs": 2,
        "log_step": 20,
        "architecture_name": "MLP",
        "dataset_name": "MNIST",
    }

    # Initialize wandb with a sample project name
    # Sweep values overwrite these defaults.
    with wandb.init(config=config_defaults) as run:
        config = run.config

        train_dataset, val_dataset = prepare_datasets(config.batch_size)

        # initialize model
        tf.keras.backend.clear_session()
        model = Model()

        # Instantiate an optimizer to train the model.
        optimizer = keras.optimizers.SGD(learning_rate=config.learning_rate)
        # Instantiate a loss function.
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # Prepare the metrics.
        metrics = [keras.metrics.SparseCategoricalAccuracy(name="acc")]

        model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
        training_history = train(
            train_dataset,
            val_dataset,
            model,
            epochs=config.epochs,
            log_step=config.log_step,
        )
        run_url = run.url

    return {"url": run_url, "epochs": len(training_history.history["loss"])}


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 7: Initialize the Sweep and Run an Agent

    The next cell registers the sweep with W&B and stores the returned
    `sweep_id`. The cell after that starts a local sweep agent.

    The agent asks W&B for hyperparameter values, calls `sweep_train` for each
    trial, and repeats until it reaches `count=10`. You can lower `count` while
    experimenting if you want a faster run.
    """)
    return


@app.cell
def _(sweep_config):
    sweep_id = wandb.sweep(sweep_config, project="sweeps-tensorflow")
    return (sweep_id,)


@app.cell
def _(sweep_id):
    wandb.agent(sweep_id, function=sweep_train, count=10)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Visualize Results

    After the sweep starts, W&B prints links to the project and sweep dashboards
    in the notebook output. Open those links to watch trials arrive live and
    compare their metrics.

    ```
    View project at:
    ```

    In the W&B UI, compare parallel coordinates, parameter importance, and the
    `epoch/val_loss` chart to understand which batch sizes and learning rates
    worked best.

    https://docs.coreweave.com/products/wandb/sweeps/visualize-sweep-results
    """)
    return


if __name__ == "__main__":
    app.run()
