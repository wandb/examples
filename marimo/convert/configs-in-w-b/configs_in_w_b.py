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

__generated_with = "0.25.1"
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

    A run's `config` records the inputs to an experiment: hyperparameters,
    dataset names, model types, and other settings. Saving these values helps
    you reproduce a run and compare experiments with different settings.
    Save outputs such as loss and accuracy with `wandb.rRun.log()`.

    This lesson creates one small example run, adds configuration values from
    Python and `argparse`, then updates the finished run with the Public API.
    It records example settings without downloading data or training a model.

    ## Authenticate with Weights & Biases

    Paste your [Forge API key](https://docs.coreweave.com/products/wandb/platform/hosting/iam/api-keys) into the password field below. Keep your key out of the notebook’s source code.

    If this runtime already has credentials in `WANDB_API_KEY` or your `.netrc` file, leave the API key field blank.

    ## Set your team and project

    Enter your Weights & Biases team name and choose a project for this lesson. The default project is `config_example`. Then select **Connect to Weights & Biases**.
    """)
    return


@app.cell(hide_code=True)
def _():
    wandb_setup_form = mo.md("{api_key}\n\n{entity}\n\n{project}").batch(
        api_key=mo.ui.text(kind="password", label="Forge API key", full_width=True),
        entity=mo.ui.text(label="Team", placeholder="Your team name", full_width=True),
        project=mo.ui.text(value="config_example", label="Project", full_width=True),
    ).form(submit_button_label="Connect to Weights & Biases", bordered=True)
    wandb_setup_form
    return (wandb_setup_form,)


@app.cell(hide_code=True)
def _(wandb_setup_form):
    mo.stop(wandb_setup_form.value is None)

    _submission = wandb_setup_form.value
    _api_key = _submission["api_key"].strip()
    _entity = _submission["entity"].strip()
    _project = _submission["project"].strip()
    mo.stop(
        not _entity or not _project,
        mo.callout("Enter your team and project, then connect again.", kind="warn"),
    )

    try:
        _login_ok = wandb.login(key=_api_key or None, relogin=bool(_api_key))
    except (wandb.errors.Error, ValueError):
        _login_ok = False

    mo.stop(
        not _login_ok,
        mo.callout("Check your API key or saved credentials, then connect again.", kind="danger"),
    )

    entity = _entity
    project = _project
    mo.md("✓ Connected to Weights & Biases.")
    return entity, project


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Start with a Python dictionary

    The dictionary below describes an example experiment using the CelebA
    dataset and a baseline model. We will pass it to `wandb.init(config=...)`
    when creating the run. Use descriptive keys with underscores or dashes
    instead of periods.
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
    active, add new fields using attribute access (`epochs`) or dictionary
    access (`batch_size`), then merge the parsed arguments with `update()`.
    You can read saved values using either style as well.

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

    saved_config
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
