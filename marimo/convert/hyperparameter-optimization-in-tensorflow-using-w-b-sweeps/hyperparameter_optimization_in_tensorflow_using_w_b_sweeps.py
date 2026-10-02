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

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="TensorFlow hyperparameter sweeps with W&B")


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
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Use Weights & Biases for machine learning experiment tracking, dataset versioning, and project collaboration.

    <img src="https://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Use Weights & Biases Sweeps to automate hyperparameter optimization and explore the space of possible models, complete with interactive dashboards like this:

    ![Example W&B Sweeps dashboard](https://i.imgur.com/AN0qnpC.png)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Why Should I Use Sweeps?

    * **Quick setup**: With just a few lines of code you can run W&B sweeps.
    * **Transparent**: We cite all the algorithms we're using, and [our code is open source](https://github.com/wandb/wandb).
    * **Powerful**: Our sweeps are completely customizable and configurable. You can launch a sweep across dozens of machines, and it's just as easy as starting a sweep on your laptop.

    **[Check out the official documentation $\rightarrow$](https://docs.wandb.ai/models/sweeps/)**

    ## What this notebook covers

    * Simple steps to get started with W&B Sweeps and TensorFlow.
    * Use Keras' current `WandbMetricsLogger` callback to track each training run.
    * Find the best hyperparameters for an image classification task.

    **Note**: Sections starting with _Step_ are all you need to perform a hyperparameter sweep in existing code.
    The rest of the code is there to set up a simple example.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Setup and Authentication

    ### Step 1: Import W&B and log in

    The setup cell imports TensorFlow, NumPy, W&B, and the dedicated
    `WandbMetricsLogger` callback. W&B's former all-in-one `WandbCallback` is
    [legacy](https://docs.wandb.ai/models/integrations/keras); new Keras
    workflows should use the dedicated metrics, checkpoint, or evaluation
    callbacks for the behavior they need. This tutorial only needs metrics.

    Enter your [W&B API key](https://wandb.ai/authorize), or leave it blank to
    use `WANDB_API_KEY` from the molab Secrets panel or credentials already
    stored in this runtime. A fresh molab session does not inherit your local
    login. If needed, [create a free W&B account](https://wandb.ai/signup).

    The entity is the team name in your W&B project URL:
    `wandb.ai/<entity>/<project>`. Leave it blank to use your default entity.
    Changing either field does nothing until you submit the form.
    """)
    return


@app.cell(hide_code=True)
def _():
    wandb_login_form = (
        mo.md("{api_key}\n\n{entity}")
        .batch(
            api_key=mo.ui.text(
                kind="password",
                label="W&B API key (optional)",
                placeholder="Paste a key or use configured credentials",
                full_width=True,
            ),
            entity=mo.ui.text(
                label="W&B entity or team (optional)",
                placeholder="Leave blank to use your default entity",
                full_width=True,
            ),
        )
        .form(submit_button_label="Connect to W&B", bordered=True)
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

    wandb_connection = {"entity": _submitted["entity"].strip() or None}
    mo.callout(mo.md("Connected to W&B."), kind="success")
    return (wandb_connection,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Prepare Dataset

    MNIST is downloaded and converted into shuffled training batches and stable
    validation batches only after you explicitly start the sweep agent. Opening
    this notebook does not download data or start training.
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
    ## Define the Model and Training Workflow

    ### Build a Simple Classifier MLP
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
    ### Write a Training Workflow

    #### Step 3: Log metrics with `WandbMetricsLogger`

    `WandbMetricsLogger` receives the training and validation metrics produced by
    `model.fit`. With `log_freq=20`, it logs batch metrics every 20 batches and
    epoch metrics at the end of each epoch. The callback is constructed only
    inside an active W&B run.
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
    ## Step 4: Configure the Sweep

    This is where you will:

    * Define the hyperparameters you're sweeping over.
    * Provide your hyperparameter optimization method. W&B supports `random`, `grid`, and `bayes` methods.
    * Provide an objective and a `metric` if using `bayes`; here we minimize `epoch/val_loss`, the validation-loss key logged by `WandbMetricsLogger`.
    * Use `hyperband` for early termination of poorly-performing runs.

    ### [Check out more on Sweep configurations $\rightarrow$](https://docs.wandb.ai/models/sweeps/define-sweep-configuration)
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
    ## Step 5: Wrap the Training Workflow

    You'll need a zero-argument function, like `sweep_train` below, that calls
    `wandb.init()` and uses the resulting run configuration before training.
    A context manager finishes every run before the agent starts its next trial.
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
    ## Step 6: Initialize the Sweep and Run an Agent

    Creating a sweep registers a new object in W&B. Enter a project and submit
    the form when you are ready; changing the unsubmitted field is inert.
    """)
    return


@app.cell(hide_code=True)
def _(sweep_config, wandb_connection):
    _entity_note = (
        f"The sweep will belong to entity `{wandb_connection['entity']}`."
        if wandb_connection["entity"]
        else "The sweep will use your default W&B entity."
    )
    _parameter_note = ", ".join(f"`{name}`" for name in sweep_config["parameters"])
    sweep_creation_form = (
        mo.md("{project}")
        .batch(
            project=mo.ui.text(
                value="sweeps-tensorflow",
                label="W&B project",
                full_width=True,
            )
        )
        .form(submit_button_label="Create W&B sweep", bordered=True)
    )
    mo.vstack(
        [
            mo.md(f"{_entity_note} This configuration searches {_parameter_note}."),
            sweep_creation_form,
        ]
    )
    return (sweep_creation_form,)


@app.cell(hide_code=True)
def _(sweep_config, sweep_creation_form, wandb_connection):
    mo.stop(
        sweep_creation_form.value is None,
        mo.callout(
            mo.md("Submit the form above when you are ready to create the sweep."),
            kind="info",
        ),
    )
    _project = sweep_creation_form.value["project"].strip()
    mo.stop(
        not _project,
        mo.callout(
            mo.md("Enter a W&B project name and submit again."),
            kind="danger",
        ),
    )
    sweep_request = {
        "config": sweep_config.copy(),
        "entity": wandb_connection["entity"],
        "project": _project,
    }
    return (sweep_request,)


@app.cell
def _(mo, sweep_request):
    sweep_entity = sweep_request["entity"] or wandb.Api().default_entity
    mo.stop(
        not sweep_entity,
        mo.callout(
            mo.md(
                "W&B did not return a default entity. Enter a team entity in "
                "the authentication form and reconnect before creating the sweep."
            ),
            kind="danger",
        ),
    )
    sweep_id = wandb.sweep(
        sweep=sweep_request["config"],
        entity=sweep_entity,
        project=sweep_request["project"],
    )
    sweep_result = {
        "id": sweep_id,
        "entity": sweep_entity,
        "project": sweep_request["project"],
        "url": (
            f"https://wandb.ai/{sweep_entity}/{sweep_request['project']}"
            f"/sweeps/{sweep_id}"
        ),
    }
    return (sweep_result,)


@app.cell(hide_code=True)
def _(sweep_result):
    mo.callout(
        mo.md(
            f"Created sweep `{sweep_result['id']}`. "
            f"[Open the live Sweep dashboard]({sweep_result['url']})."
        ),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _(sweep_result, sweep_train):
    _training_function = sweep_train.__name__
    sweep_agent_form = (
        mo.md(
            f"Run an agent for sweep `{sweep_result['id']}`. The original "
            "tutorial uses 10 trials; lower the count for a quicker test."
            f" Each trial calls `{_training_function}`.\n\n{{count}}"
        )
        .batch(
            count=mo.ui.number(
                start=1,
                stop=10,
                step=1,
                value=10,
                label="Maximum sweep trials",
            )
        )
        .form(submit_button_label="Start sweep agent and training", bordered=True)
    )
    sweep_agent_form
    return (sweep_agent_form,)


@app.cell(hide_code=True)
def _(sweep_agent_form, sweep_result, sweep_train):
    mo.stop(
        sweep_agent_form.value is None,
        mo.callout(
            mo.md(
                "Submit the form above to download MNIST and start the requested "
                "number of W&B training runs."
            ),
            kind="info",
        ),
    )
    agent_request = {
        "count": int(sweep_agent_form.value["count"]),
        "function": sweep_train,
        "sweep": sweep_result.copy(),
    }
    return (agent_request,)


@app.cell
def _(agent_request):
    if wandb.run is not None:
        wandb.run.finish()

    _sweep = agent_request["sweep"]
    wandb.agent(
        _sweep["id"],
        function=agent_request["function"],
        entity=_sweep["entity"],
        project=_sweep["project"],
        count=agent_request["count"],
    )
    agent_result = {
        "count": agent_request["count"],
        "url": _sweep["url"],
    }
    return (agent_result,)


@app.cell(hide_code=True)
def _(agent_result):
    mo.callout(
        mo.md(
            f"The agent completed up to {agent_result['count']} trials. "
            f"[Open the Sweep dashboard]({agent_result['url']}) to compare "
            "`epoch/val_loss`, batch size, and learning rate across runs."
        ),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Visualize Results

    After creating the sweep, open the **Sweep dashboard** link above to see
    results arrive live. Compare parallel coordinates, parameter importance,
    and the `epoch/val_loss` chart to understand which configurations work best.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Example Gallery

    See examples of projects tracked and visualized with W&B in our [Gallery →](https://wandb.ai/gallery).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Best Practices

    1. **Projects**: Log multiple runs to a project to compare them. `wandb.init(project="project-name")`
    2. **Groups**: For multiple processes or cross validation folds, log each process as a run and group them together. `wandb.init(group="experiment-1")`
    3. **Tags**: Add tags to track your current baseline or production model.
    4. **Notes**: Type notes in the table to track the changes between runs.
    5. **Reports**: Take quick notes on progress to share with colleagues and make dashboards and snapshots of your ML projects.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Advanced Setup

    1. [Environment variables](https://docs.wandb.ai/models/track/environment-variables): Set API keys in environment variables so you can run training on a managed cluster.
    2. [Offline mode](https://docs.wandb.ai/models/ref/cli/wandb-offline): Set `WANDB_MODE=offline` to train without syncing, then upload the saved run later with `wandb sync`.
    3. [Self-managed W&B](https://docs.wandb.ai/guides/hosting/): Run W&B in your own infrastructure, including private-cloud and air-gapped environments.
    """)
    return


if __name__ == "__main__":
    app.run()
