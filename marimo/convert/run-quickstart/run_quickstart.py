# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "wandb",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="W&B Run Quickstart")

with app.setup:
    import random

    import marimo as mo
    import wandb


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

    # Weights & Biases Run Quickstart

    A Weights & Biases run represents one experiment. It records your configuration, metrics,
    and results so you can inspect an experiment or compare it with other runs.

    This notebook simulates training to focus on three Weights & Biases operations:
    `wandb.init()` starts a run, `training_run.log()` records metrics, and
    `wandb.Api()` retrieves the results.

    Running the code cells creates a Weights & Biases run and uploads its simulated metrics.
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
            "Run the next cell to verify them with W&B.",
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
    credentials and load your available W&B entities.
    """)
    return


@app.cell(hide_code=True)
def _():
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
                "W&B verified the credentials but could not retrieve your account entities.",
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
            "W&B authenticated successfully, but no personal or team entities were found.",
            kind="danger",
        ),
    )

    mo.callout(
        mo.md("✓ Credentials verified. Choose a personal or team entity below."),
        kind="success",
    )
    return (available_entities,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Set your team and project

    Run the following cell and specify your:

    - Weights & Biases personal or team entity.
    - A name for your [project](https://docs.coreweave.com/products/wandb/track/project-page#projects). The default project is `my-awesome-project`.

    Then select **Use this W&B target**.

    > If the project doesn't exist, Weights & Biases creates it when the first run starts, provided you have write access.
    """)
    return


@app.cell(hide_code=True)
def _():
    def _validate_wandb_target(value):
        if value is None:
            return None
        if not value["entity"]:
            return "Select a W&B entity."
        if not value["project"].strip():
            return "Enter a project name."
        return None

    def _notify_wandb_submission(_value):
        mo.status.toast(
            "W&B target selected",
            "Run the next cell to confirm the destination.",
        )

    wandb_setup_form = (
        mo.md("{entity}\n\n{project}")
        .batch(
            entity=mo.ui.dropdown(
                options=available_entities,
                value=available_entities[0],
                label="W&B entity",
                searchable=True,
                full_width=True,
            ),
            project=mo.ui.text(
                value="my-awesome-project",
                label="Project",
                full_width=True,
            ),
        )
        .form(
            submit_button_label="Use this W&B target",
            validate=_validate_wandb_target,
            on_change=_notify_wandb_submission,
            bordered=True,
        )
    )

    wandb_setup_form
    return (wandb_setup_form,)




@app.cell(hide_code=True)
def _():
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
    ## Define the Experiment Configuration

    The next cell defines a configuration dictionary. Passing it to
    `wandb.init(config=config)` records these settings with the run so you can
    compare experiments by hyperparameter.

    We simulate ten epochs with a fixed random seed. The learning rate is
    recorded to demonstrate configuration tracking. It does not affect our
    synthetic metrics. In real training, you would use it to configure an
    optimizer.
    """)
    return


@app.cell
def _():
    config = {
        "epochs": 10,
        "lr": 0.01,
        "seed": 42,
    }
    return (config,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Start a Run and Log Metrics

    The next cell starts `training_run` with `wandb.init()`. Inside the loop,
    `training_run.log()` records a dictionary containing the epoch, accuracy,
    and loss. In your own training code, replace the simulated values with
    measured metrics.

    The context manager finishes the run and completes logging when the block
    exits. We keep its URL for the dashboard and its
    `entity/project/run_id` path for the API query that follows.
    """)
    return


@app.cell
def _(config, entity, project):
    rng = random.Random(config["seed"])
    offset = rng.random() / 5

    with wandb.init(project=project, entity=entity, config=config) as training_run:
        for epoch in range(1, config["epochs"] + 1):
            accuracy = 1 - 2 ** -epoch - rng.random() / config["epochs"] - offset
            loss = 2 ** -epoch + rng.random() / config["epochs"] + offset
            training_run.log({
                "epoch": epoch,
                "accuracy": accuracy,
                "loss": loss,
            })
            print(f"Epoch {epoch}: accuracy={accuracy:.4f}, loss={loss:.4f}")

        # For demo: Store run path to retrieve run info (see following cell)
        run_path = f"{training_run.entity}/{training_run.project}/{training_run.id}"
    return (run_path,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Retrieve the Logged Results

    The next cell uses `wandb.Api().run(run_path)` to fetch the completed
    experiment. `logged_run.config` contains its settings, and
    `logged_run.summary` includes the latest logged metric values.

    `scan_history()` iterates over every recorded row containing the requested
    keys. We collect those rows in a list so we can inspect the complete epoch
    history, rather than a sampled subset.
    """)
    return


@app.cell
def _(run_path):
    api = wandb.Api()
    logged_run = api.run(run_path)
    history = list(logged_run.scan_history(keys=["epoch", "accuracy", "loss"]))
    history
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Verify the logged run

    Open the run URL printed by `wandb.init()`. Confirm that `accuracy` rises and
    `loss` falls across the logged epochs. Open **Config** and compare it with the
    notebook dictionary, then check **Summary** for the latest values.

    The Public API provides a second check: `scan_history()` should contain every
    logged epoch, while `logged_run.summary` contains the latest values. See the
    [config guide](https://docs.coreweave.com/products/wandb/track/config#view-config-values)
    and [summary guide](https://docs.coreweave.com/products/wandb/track/log/log-summary#view-summary-metrics)
    for those views.

    ## Next steps

    Change the seed or epoch count and rerun to create another run. Compare complete
    histories rather than only final values. In real training, replace simulated
    metrics with measured results and record the learning rate in Config so the
    comparison has an interpretable cause.
    """)
    return


if __name__ == "__main__":
    app.run()
