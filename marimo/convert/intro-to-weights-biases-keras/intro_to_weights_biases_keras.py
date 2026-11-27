# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "marimo>=0.24.2",
#     "tensorflow>=2.16,<3",
#     "wandb>=0.18,<1",
# ]
# ///
"""Learn the W&B experiment-tracking workflow with simulated and Keras runs."""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="Introduction to W&B with Keras")


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
    # Introduction to Weights & Biases with Keras

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/intro-to-weights-biases-keras/intro_to_weights_biases_keras.py/server)

    <img src="https://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    Use [Weights & Biases](https://wandb.ai/site?utm_source=keras_intro_colab&utm_medium=code&utm_campaign=keras_intro)
    for experiment tracking, model checkpointing, and collaboration. This
    tutorial starts with simulated metrics, then trains a small Keras classifier,
    and finishes with a scriptable alert.

    Nothing is sent to W&B, no dataset is downloaded, and no model is trained
    until you explicitly connect and submit the corresponding action form.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## A shared dashboard for your experiments

    With just a few lines of code, you can create rich, interactive, shareable
    dashboards like this [example W&B project](https://wandb.ai/wandb/wandb_example).

    ![Example W&B dashboard](https://i.imgur.com/Pell4Oo.png)

    ## Data and privacy

    W&B encrypts data in transit and at rest. Organizations that need to keep
    data in their own environment can use
    [W&B Self-Managed](https://docs.wandb.ai/platform/hosting/hosting-options/self-managed).
    You can also query and export logged data with the
    [W&B Public API](https://docs.wandb.ai/models/ref/python/public-api/).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Connect to W&B

    Enter your [W&B API key](https://wandb.ai/authorize), or leave it blank to
    use `WANDB_API_KEY` from the molab Secrets panel or credentials already
    configured in this runtime. A new molab session does not inherit a login
    from your computer.

    The entity is the team name in a W&B project URL:
    `wandb.ai/<entity>/<project>`. Leave it blank to use your default entity.
    Editing either field is inert until you submit the form.
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
            mo.md("Submit the form above before starting any W&B run."),
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
    ## Run a simulated experiment

    The basic W&B workflow has three steps:

    1. Start a run and record the hyperparameters in its config.
    2. Log metrics from a training or evaluation loop.
    3. Open the run links to compare results in the dashboard.

    The original tutorial launched five runs. Choose a smaller count for a quick
    test or keep five for the full comparison. Submitting the form creates that
    many remote runs in the selected project.
    """)
    return


@app.cell(hide_code=True)
def _():
    quickstart_form = (
        mo.md("{project}\n\n{run_count}\n\n{seed}")
        .batch(
            project=mo.ui.text(
                value="basic-intro",
                label="W&B project",
                full_width=True,
            ),
            run_count=mo.ui.number(
                start=1,
                stop=5,
                step=1,
                value=5,
                label="Number of simulated runs",
            ),
            seed=mo.ui.number(
                start=0,
                stop=10_000,
                step=1,
                value=42,
                label="Random seed",
            ),
        )
        .form(submit_button_label="Create simulated W&B runs", bordered=True)
    )
    quickstart_form
    return (quickstart_form,)


@app.cell(hide_code=True)
def _(quickstart_form, wandb_connection):
    mo.stop(
        quickstart_form.value is None,
        mo.callout(
            mo.md("Submit the form to create and log the simulated runs."),
            kind="info",
        ),
    )

    _submitted = quickstart_form.value
    _project = _submitted["project"].strip() or "basic-intro"
    _rng = random.Random(int(_submitted["seed"]))
    _run_links = []

    # Launch simulated experiments. Each context manager finishes its run.
    for _index in range(int(_submitted["run_count"])):
        _config = {
            "learning_rate": 0.02,
            "architecture": "CNN",
            "dataset": "CIFAR-100",
        }
        with wandb.init(
            entity=wandb_connection["entity"],
            project=_project,
            name=f"simulated-run-{_index + 1}",
            config=_config,
        ) as _run:
            _offset = _rng.random() / 5
            for _step in range(2, 10):
                _accuracy = 1 - 2**(-_step) - _rng.random() / _step - _offset
                _loss = 2**(-_step) + _rng.random() / _step + _offset
                # Log metrics from the simulated training loop.
                _run.log({"accuracy": _accuracy, "loss": _loss})
            _run_links.append((_run.name, _run.url))

    quickstart_result = {
        "project": _project,
        "runs": tuple(_run_links),
    }
    return (quickstart_result,)


@app.cell(hide_code=True)
def _(quickstart_result):
    _links = "\n".join(
        f"- [{_name}]({_url})" if _url else f"- {_name} (URL unavailable)"
        for _name, _url in quickstart_result["runs"]
    )
    mo.callout(
        mo.md(
            f"Created {len(quickstart_result['runs'])} runs in "
            f"`{quickstart_result['project']}`:\n\n{_links}"
        ),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Train a simple Keras classifier

    This example downloads MNIST, trains a small dense classifier, and streams
    metrics to W&B. It uses the current dedicated callbacks:

    - `WandbMetricsLogger` records Keras training and validation metrics.
    - `WandbModelCheckpoint` saves the best model and logs it as a W&B Artifact.

    The former all-in-one `WandbCallback` is
    [legacy](https://docs.wandb.ai/models/integrations/keras); new workflows
    should select the dedicated callback for each behavior they need.

    Checkpoints are written to a temporary directory and removed after W&B has
    finished the run, so rerunning this notebook does not leave model files in
    the repository.
    """)
    return


@app.cell(hide_code=True)
def _():
    keras_form = (
        mo.md("{project}\n\n{experiment_count}\n\n{epochs}\n\n{batch_size}\n\n{seed}")
        .batch(
            project=mo.ui.text(
                value="keras-intro",
                label="W&B project",
                full_width=True,
            ),
            experiment_count=mo.ui.number(
                start=1,
                stop=5,
                step=1,
                value=5,
                label="Number of Keras experiments",
            ),
            epochs=mo.ui.number(
                start=1,
                stop=10,
                step=1,
                value=6,
                label="Epochs per experiment",
            ),
            batch_size=mo.ui.dropdown(
                options=[128, 256, 512],
                value=256,
                label="Batch size",
            ),
            seed=mo.ui.number(
                start=0,
                stop=10_000,
                step=1,
                value=42,
                label="Random seed",
            ),
        )
        .form(
            submit_button_label="Download MNIST and train Keras experiments",
            bordered=True,
        )
    )
    keras_form
    return (keras_form,)


@app.cell(hide_code=True)
def _(keras_form, wandb_connection):
    mo.stop(
        keras_form.value is None,
        mo.callout(
            mo.md("Submit the form to download MNIST and start training."),
            kind="info",
        ),
    )

    _submitted = keras_form.value
    _project = _submitted["project"].strip() or "keras-intro"
    _seed = int(_submitted["seed"])
    _rng = random.Random(_seed)

    # Download and normalize MNIST only after explicit form submission.
    (_x_train, _y_train), (_x_test, _y_test) = tf.keras.datasets.mnist.load_data()
    _x_train, _x_test = _x_train / 255.0, _x_test / 255.0
    _x_train, _y_train = _x_train[::5], _y_train[::5]
    _x_test, _y_test = _x_test[::20], _y_test[::20]

    _run_summaries = []
    with tempfile.TemporaryDirectory(prefix="wandb-keras-checkpoints-") as _checkpoint_dir:
        for _index in range(int(_submitted["experiment_count"])):
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(_seed + _index)
            _config = {
                "layer_1": 512,
                "activation_1": "relu",
                "dropout": _rng.uniform(0.01, 0.80),
                "layer_2": 10,
                "activation_2": "softmax",
                "optimizer": "sgd",
                "loss": "sparse_categorical_crossentropy",
                "metric": "accuracy",
                "epochs": int(_submitted["epochs"]),
                "batch_size": int(_submitted["batch_size"]),
            }

            with wandb.init(
                entity=wandb_connection["entity"],
                project=_project,
                name=f"keras-mnist-{_index + 1}",
                config=_config,
            ) as _run:
                _model = tf.keras.Sequential(
                    [
                        tf.keras.Input(shape=(28, 28)),
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(
                            _run.config["layer_1"],
                            activation=_run.config["activation_1"],
                        ),
                        tf.keras.layers.Dropout(_run.config["dropout"]),
                        tf.keras.layers.Dense(
                            _run.config["layer_2"],
                            activation=_run.config["activation_2"],
                        ),
                    ]
                )
                _model.compile(
                    optimizer=_run.config["optimizer"],
                    loss=_run.config["loss"],
                    metrics=[_run.config["metric"]],
                )

                _checkpoint_path = str(
                    Path(_checkpoint_dir)
                    / f"experiment-{_index + 1}-best.keras"
                )
                _history = _model.fit(
                    x=_x_train,
                    y=_y_train,
                    epochs=_run.config["epochs"],
                    batch_size=_run.config["batch_size"],
                    validation_data=(_x_test, _y_test),
                    callbacks=[
                        WandbMetricsLogger(log_freq="epoch"),
                        WandbModelCheckpoint(
                            filepath=_checkpoint_path,
                            monitor="val_loss",
                            mode="min",
                            save_best_only=True,
                        ),
                    ],
                    verbose=2,
                )
                _run_summaries.append(
                    {
                        "name": _run.name,
                        "url": _run.url,
                        "val_accuracy": float(
                            _history.history["val_accuracy"][-1]
                        ),
                    }
                )

    keras_result = {
        "project": _project,
        "runs": tuple(_run_summaries),
    }
    return (keras_result,)


@app.cell(hide_code=True)
def _(keras_result):
    _rows = "\n".join(
        (
            f"- [{_run['name']}]({_run['url']}): "
            f"final validation accuracy `{_run['val_accuracy']:.3f}`"
            if _run["url"]
            else (
                f"- {_run['name']}: final validation accuracy "
                f"`{_run['val_accuracy']:.3f}` (URL unavailable)"
            )
        )
        for _run in keras_result["runs"]
    )
    mo.callout(
        mo.md(
            f"Finished {len(keras_result['runs'])} Keras experiments in "
            f"`{keras_result['project']}`:\n\n{_rows}"
        ),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Try W&B Alerts

    [W&B Alerts](https://docs.wandb.ai/models/runs/alert) can notify you through
    Slack or email when code detects a condition such as low accuracy. Before
    trying this section, enable **Scriptable run alerts** and configure a
    destination in your W&B user settings.

    The demo logs simulated accuracy values and calls `run.alert()` on the first
    value at or below the threshold. Submitting this form creates one run and
    may send a real external notification.
    """)
    return


@app.cell(hide_code=True)
def _():
    alert_form = (
        mo.md("{project}\n\n{threshold}\n\n{max_steps}\n\n{seed}")
        .batch(
            project=mo.ui.text(
                value="keras-intro",
                label="W&B project",
                full_width=True,
            ),
            threshold=mo.ui.number(
                start=0.0,
                stop=2.0,
                step=0.05,
                value=0.3,
                label="Low-accuracy threshold",
            ),
            max_steps=mo.ui.number(
                start=1,
                stop=1_000,
                step=1,
                value=1_000,
                label="Maximum simulated steps",
            ),
            seed=mo.ui.number(
                start=0,
                stop=10_000,
                step=1,
                value=7,
                label="Random seed",
            ),
        )
        .form(
            submit_button_label="Run demo and send alert if triggered",
            bordered=True,
        )
    )
    alert_form
    return (alert_form,)


@app.cell(hide_code=True)
def _(alert_form, wandb_connection):
    mo.stop(
        alert_form.value is None,
        mo.callout(
            mo.md("Submit the form to create the alert demo run."),
            kind="info",
        ),
    )

    _submitted = alert_form.value
    _project = _submitted["project"].strip() or "keras-intro"
    _threshold = float(_submitted["threshold"])
    _rng = random.Random(int(_submitted["seed"]))
    _trigger = None

    with wandb.init(
        entity=wandb_connection["entity"],
        project=_project,
        name="low-accuracy-alert-demo",
    ) as _run:
        for _step in range(int(_submitted["max_steps"])):
            _accuracy = round(_rng.random() + _rng.random(), 3)
            _run.log({"accuracy": _accuracy})

            if _accuracy <= _threshold:
                _run.alert(
                    title="Low accuracy",
                    text=(
                        f"Accuracy {_accuracy} at step {_step} is below the "
                        f"acceptable threshold {_threshold}."
                    ),
                    level=AlertLevel.WARN,
                    wait_duration=300,
                )
                _trigger = {"step": _step, "accuracy": _accuracy}
                break
        _run_url = _run.url

    alert_result = {
        "run_url": _run_url,
        "trigger": _trigger,
        "max_steps": int(_submitted["max_steps"]),
    }
    return (alert_result,)


@app.cell(hide_code=True)
def _(alert_result):
    if alert_result["trigger"] is None:
        _message = (
            f"No accuracy value crossed the threshold in "
            f"{alert_result['max_steps']} steps, so no alert was sent."
        )
    else:
        _message = (
            f"Sent the alert at step {alert_result['trigger']['step']} with "
            f"accuracy `{alert_result['trigger']['accuracy']}`."
        )

    _run_link = (
        f"[Open the alert demo run]({alert_result['run_url']})."
        if alert_result["run_url"]
        else "The run URL is unavailable."
    )
    mo.callout(mo.md(f"{_message}\n\n{_run_link}"), kind="success")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What's next?

    Continue with
    [Organizing hyperparameter sweeps in PyTorch](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/organizing-hyperparameter-sweeps-in-pytorch-with-w-b/organizing_hyperparameter_sweeps_in_pytorch_with_w_b.py/server)
    to automate a collection of training runs.
    """)
    return


if __name__ == "__main__":
    app.run()
