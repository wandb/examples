# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "marimo>=0.24.2",
#     "wandb==0.30.0",
# ]
#
# [tool.marimo.runtime]
# on_cell_change = "lazy"
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(auto_download=["html"])

with app.setup:
    import argparse
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

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/configs-in-w-b/configs_in_w_b.py/server)

    # Save experiment configurations with Weights & Biases

    A [run's `config`](https://docs.coreweave.com/products/wandb/track/config) records the settings you supply for an experiment, such as
    hyperparameters, dataset names, and model types. Saving these values helps
    you reproduce a run and compare experiments with different settings.
    Save outputs such as loss and accuracy with [`wandb.Run.log()`](https://docs.coreweave.com/products/wandb/ref/python/experiments/run#method-run-log).

    This notebook creates one example run, adds configuration values from
    Python and `argparse`, then updates the finished run with the Public API.

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
    - A name for your [project](https://docs.coreweave.com/products/wandb/track/project-page#projects). The default project is `config_example`.

    Then select **Use this W&B target**. A project does not need to exist yet;
    W&B creates it when the first run starts if you have write access.
    """)
    return


@app.cell(hide_code=True)
def _(available_entities):
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
                value="config_example",
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
    mo.md(r"""
    After submitting the target form, run the next cell to confirm the selected
    destination. W&B verifies project write access when the first run starts.
    """)
    return


@app.cell(hide_code=True)
def _(wandb_setup_form):
    mo.stop(wandb_setup_form.value is None)

    _submission = wandb_setup_form.value
    _entity = _submission["entity"]
    _project = _submission["project"].strip()

    entity = _entity
    project = _project

    mo.callout(
        mo.md(
            f"✓ Using `{entity}/{project}`. W&B will confirm write access "
            "when the first run starts."
        ),
        kind="success",
    )
    return entity, project


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Start with a Python dictionary

    The dictionary below records sample values: `CelebA` as a dataset name and `baseline` as a model type. These are example metadata values, not real objects.

    These sample values are passed to `wandb.init(config=...)` when creating the run. This enables us to keep track of what the run uses during training.
    """)
    return


@app.cell
def _():
    config = {
        "dataset": "CelebA",
        "type": "baseline",
    }
    return (config,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Collect command-line settings with `argparse`

    Training scripts often receive hyperparameters as command-line arguments.
    Weights & Biases accepts the `argparse.Namespace` returned by `parse_args()` directly
    in `run.config.update()`.

    Here, `parse_args([])` selects the defaults without reading the notebook
    process's arguments. Try `parse_args(["--batch_per_gpu", "16"])` to see
    how an override changes the saved configuration. In a standalone training
    script, use `parse_args()` to read arguments from the command line.
    """)
    return


@app.cell
def _():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_per_gpu", type=int, default=8)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.1)
    args = parser.parse_args([])
    args
    return (args,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Save and extend the run's configuration

    `wandb.init(config=config)` saves the initial dictionary. While the run is
    active, add new fields using attribute access or dictionary
    access, then merge the parsed arguments with `update()`.

    The cell displays the combined configuration so you can check all six
    fields. The context manager finishes and syncs the run before the next
    example queries it. It also saves the actual entity, project, and run ID
    in `run_path`, so no ID needs to be copied by hand.
    """)
    return


@app.cell
def _(args, config, entity, project):
    with wandb.init(entity=entity, project=project, config=config) as run:
        run.config.epochs = 4
        run.config["batch_size"] = 32
        run.config.update(args)
        run_path = f"{run.entity}/{run.project}/{run.id}"
        saved_config = dict(run.config)

    print(saved_config)
    return (run_path,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Add metadata after a run has finished

    If you forgot to record a setting, use the
    [Weights & Biases Public API](https://docs.coreweave.com/products/wandb/ref/python/public-api)
    to retrieve the completed run at `entity/project/run_id`. This example
    uses the `run_path` captured above and adds a `purpose` field to that run.
    Calling `update()` saves the edited configuration to Weights & Biases.
    """)
    return


@app.cell
def _(run_path):
    api = wandb.Api()
    completed_run = api.run(run_path)
    completed_run.config["purpose"] = "configuration tutorial"
    completed_run.update()
    completed_run.config
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Verify and compare experiments

    Open the run link printed by `wandb.init()` and find **Config** on the
    run's **Overview** page. Confirm that it contains `dataset="CelebA"`,
    `type="baseline"`, `epochs=4`, `batch_size=32`, `batch_per_gpu=8`, and
    `weight_decay=0.1` (or your chosen overrides). Refresh after the Public API
    step to see `purpose="configuration tutorial"` as well.

    Rerun the example with different settings, then open the project's runs
    table to filter or group by a config field such as `type` or `batch_per_gpu`.
    In a training experiment, log metrics with `run.log()` to compare outcomes
    across configurations or build a parallel coordinates chart.

    Explore the [example project](https://wandb.ai/wandb/DistHyperOpt) for a
    larger comparison, or read
    [Configure experiments](https://docs.coreweave.com/products/wandb/track/config)
    for more configuration patterns.
    """)
    return


if __name__ == "__main__":
    app.run()
