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

    This lesson simulates training to focus on three Weights & Biases operations:
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

    If this runtime already has credentials in `WANDB_API_KEY` or your `.netrc` file, leave the API key field blank.

    ## Set your team and project

    Enter your Weights & Biases team name and choose a project for this lesson. The default project is `my-awesome-project`. Then select **Connect to Weights & Biases**.
    """)
    return


@app.cell(hide_code=True)
def _():
    wandb_setup_form = mo.md("{api_key}\n\n{entity}\n\n{project}").batch(
        api_key=mo.ui.text(kind="password", label="Forge API key", full_width=True),
        entity=mo.ui.text(label="Team", placeholder="Your team name", full_width=True),
        project=mo.ui.text(value="my-awesome-project", label="Project", full_width=True),
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
    ## Explore the Run in Weights & Biases

    Open the Weights & Biases run to view the accuracy and loss charts. Its
    configuration contains the settings from this notebook, and its summary
    shows the most recently logged metric values.

    Change the seed or epoch count and rerun the experiment. Weights & Biases creates a new
    run that you can compare with the previous one in your project. In a real
    training workflow, changing the learning rate would let you compare how that
    hyperparameter affects measured performance.

    See the [Weights & Biases quickstart](https://docs.wandb.ai/models/quickstart)
    for more examples.
    """)
    return


if __name__ == "__main__":
    app.run()
