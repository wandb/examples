# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "marimo>=0.24.2",
#     "torch>=2.1,<3",
#     "torchvision>=0.16,<1",
#     "wandb>=0.18,<1",
# ]
# ///
"""Organize a PyTorch hyperparameter search with Weights & Biases Sweeps."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    app_title="PyTorch hyperparameter sweeps with Weights & Biases",
)

with app.setup:
    import pprint
    import tempfile
    from pathlib import Path

    import marimo as mo
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torchvision import datasets, transforms

    import wandb

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    data_root = Path(tempfile.gettempdir()) / "wandb-pytorch-sweeps-mnist"


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

    # Organizing Hyperparameter Sweeps in PyTorch with Weights & Biases

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/organizing-hyperparameter-sweeps-in-pytorch-with-w-b/organizing_hyperparameter_sweeps_in_pytorch_with_w_b.py/server)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Comparing hyperparameters requires repeated training runs. A sweep defines
    which settings to vary and which metric to compare, then records the
    configuration and results of each trial.

    This example samples optimizer, hidden-layer size, dropout, learning rate,
    and batch size for a small MNIST classifier. It minimizes training loss to
    demonstrate sweep mechanics. Use a validation metric when selecting settings
    for a model you intend to deploy.

    In this tutorial you will create a hyperparameter search with Weights & Biases PyTorch integration. Follow along with a [video tutorial](https://wandb.me/sweeps-video).

    ![](https://i.imgur.com/WVKkMWw.png)
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
    - A name for your [project](https://docs.coreweave.com/products/wandb/track/project-page#projects). The default project is `pytorch-sweeps-demo`.

    Then select **Use this Weights & Biases target**.

    > If the project doesn't exist, Weights & Biases creates it when the first run starts, provided you have write access.
    """)
    return


@app.cell(hide_code=True)
def _():
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
                value="pytorch-sweeps-demo",
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
def _():
    mo.md(r"""
    After submitting the target form, run the next cell to confirm the selected
    destination. Weights & Biases verifies project write access when the first run starts.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.stop(wandb_setup_form.value is None)

    _submission = wandb_setup_form.value
    _entity = _submission["entity"]
    _project = _submission["project"].strip()

    entity = _entity
    project = _project

    mo.callout(
        mo.md(
            f"✓ Using `{entity}/{project}`. Weights & Biases will confirm write access "
            "when the first run starts."
        ),
        kind="success",
    )
    return entity, project




@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Define a sweep

    The next cell defines the complete search space in one nested dictionary.
    Weights & Biases supports `grid`, `random`, and `bayes` search.

    The objective of the hyperparameter search in this notebook is to minimize the metric named `loss`, which the training function will log once per epoch.

    Under `parameters`, use:

    - `values` for equally likely choices: optimizer, hidden-layer size, and dropout.
    - `value` for a fixed setting: one epoch per trial.
    - `distribution` for a sampled range: a uniform learning rate between 0 and
      0.1, and a logarithmically sampled batch size between 32 and 256, rounded
      to a multiple of 8 by `q`.

    See the [sweep configuration reference](https://docs.wandb.ai/models/sweeps/sweep-config-keys) for the full set of options. The command-line equivalent uses a YAML file.
    """)
    return


@app.cell
def _():
    sweep_config = {
        "method": "random",
        "metric": {"name": "loss", "goal": "minimize"},
        "parameters": {
            "optimizer": {"values": ["adam", "sgd"]},
            "fc_layer_size": {"values": [128, 256, 512]},
            "dropout": {"values": [0.3, 0.4, 0.5]},
            "epochs": {"value": 1},
            "learning_rate": {
                "distribution": "uniform",
                "min": 0,
                "max": 0.1,
            },
            "batch_size": {
                "distribution": "q_log_uniform_values",
                "q": 8,
                "min": 32,
                "max": 256,
            },
        },
    }
    pprint.pprint(sweep_config)
    return (sweep_config,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Initialize the sweep

    `wandb.sweep()` registers the configuration with a Weights & Biases-managed controller
    and returns `sweep_id`. The controller chooses hyperparameters. An agent on
    this machine requests those values and executes the training function.
    Rerunning this cell creates a new sweep. Weights & Biases prints its dashboard link.

    In a terminal, register the same configuration with `wandb sweep config.yaml`.
    See the [Sweeps guide](https://docs.wandb.ai/models/sweeps/) for details.
    """)
    return


@app.cell
def _(entity, project, sweep_config):
    sweep_id = wandb.sweep(sweep=sweep_config, entity=entity, project=project)
    return (sweep_id,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Define the PyTorch training code

    The following helpers handle the repeated work in each trial:

    - `build_dataset` normalizes MNIST images and takes every fifth training
      example to keep trials short.
    - `build_network` creates a fully connected classifier with a configurable
      hidden-layer size and dropout, then moves it to the available device.
    - `build_optimizer` selects Adam or SGD with the sampled learning rate.
    - `train_epoch` performs forward and backward passes and logs `batch loss`.
      It weights each batch loss by its number of examples so the epoch's mean
      stays comparable when the sweep changes batch size.

    This notebook optimizes mean training loss. For model selection in your own
    project, log a held-out validation metric and use it as the sweep objective.
    """)
    return


@app.function
def build_dataset(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset = datasets.MNIST(
        data_root, train=True, download=True, transform=transform
    )
    sub_dataset = torch.utils.data.Subset(
        dataset, indices=range(0, len(dataset), 5)
    )
    return torch.utils.data.DataLoader(sub_dataset, batch_size=batch_size)


@app.function
def build_network(fc_layer_size, dropout):
    network = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, fc_layer_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(fc_layer_size, 10),
        nn.LogSoftmax(dim=1),
    )
    return network.to(device)


@app.function
def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        return optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)
    if optimizer == "adam":
        return optim.Adam(network.parameters(), lr=learning_rate)
    raise ValueError(f"Unknown optimizer: {optimizer}")


@app.function
def train_epoch(sweep_run, network, loader, optimizer):
    network.train()
    total_loss = 0.0
    total_examples = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(network(data), target)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss * data.size(0)
        total_examples += data.size(0)
        sweep_run.log({"batch loss": batch_loss})

    return total_loss / total_examples


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The sweep agent needs a zero-argument function to call for each trial. `train()`
    starts `sweep_run` with `wandb.init()` and reads the controller's sampled
    hyperparameters from `sweep_run.config`. It builds fresh data, model, and
    optimizer objects, then logs the epoch's mean loss under the exact `loss`
    key named in `sweep_config`.

    The run name identifies its optimizer and hidden-layer size, with the run ID
    to distinguish trials. Exiting the context manager finishes each run before
    the agent starts the next one.
    """)
    return


@app.function
def train():
    with wandb.init(job_type="sweep-trial") as sweep_run:
        config = sweep_run.config
        sweep_run.name = (
            f"mnist-{config.optimizer}-h{config.fc_layer_size}-{sweep_run.id}"
        )
        loader = build_dataset(config.batch_size)
        network = build_network(config.fc_layer_size, config.dropout)
        optimizer = build_optimizer(network, config.optimizer, config.learning_rate)

        for epoch in range(config.epochs):
            avg_loss = train_epoch(sweep_run, network, loader, optimizer)
            sweep_run.log({"loss": avg_loss, "epoch": epoch})


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Run a sweep agent

    Set `trial_count` below, then run the agent cell. `wandb.agent()` asks the
    controller for hyperparameters and calls `train()` up to that many times.
    Lower the count for a quicker experiment. Rerunning the agent cell adds
    trials to the existing sweep.

    You can run additional agents with the same `sweep_id` on other machines.
    The controller coordinates their trials using the sweep configuration.
    Because this sweep uses random search, each requested configuration is
    sampled from the search space above.
    """)
    return


@app.cell
def _():
    trial_count = 5
    return (trial_count,)


@app.cell
def _(entity, project, sweep_id, trial_count):
    wandb.agent(
        sweep_id,
        function=train,
        entity=entity,
        project=project,
        count=trial_count,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Verify and visualize sweep results

    Open the sweep dashboard link printed by Weights & Biases. Confirm that completed runs
    contain the sampled configuration, a `batch loss` chart, and an epoch-level
    `loss` chart. Sort runs by `loss` to find the lowest training loss. Five
    trials demonstrate the workflow. Larger searches provide more evidence for
    the comparison plots below.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Parallel Coordinates Plot
    This plot maps hyperparameter values to model metrics. It’s useful for honing in on combinations of hyperparameters that led to the best model performance.

    ![](https://assets.website-files.com/5ac6b7f2924c652fd013a891/5e190366778ad831455f9af2_s_194708415DEC35F74A7691FF6810D3B14703D1EFE1672ED29000BA98171242A5_1578695138341_image.png)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Hyperparameter Importance Plot
    The hyperparameter importance plot estimates which settings best predict
    the observed metric values. Use it to choose settings worth investigating
    further, rather than treating it as proof that a parameter caused an improvement.
    With only a few trials, these estimates can be unstable.

    ![](https://assets.website-files.com/5ac6b7f2924c652fd013a891/5e190367778ad820b35f9af5_s_194708415DEC35F74A7691FF6810D3B14703D1EFE1672ED29000BA98171242A5_1578695757573_image.png)

    Compare these patterns with individual runs before narrowing the search
    space. Repeat promising settings to check whether the improvement persists.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Verify and interpret the sweep

    Open the URL printed by `wandb.sweep()`; it ends in
    `/sweeps/<sweep-id>`. Add `optimizer`, `learning_rate`, `fc_layer_size`,
    `dropout`, and the objective `loss` to the runs table. Each trial should also
    contain `batch loss` and epoch-level `loss` charts.

    Sort by `loss`, then inspect parallel coordinates and parameter importance.
    Use importance as a hypothesis generator, especially with few trials. This
    tutorial minimizes training loss, so the lowest run is not automatically the
    model that generalizes best. See the
    [Sweeps visualization guide](https://docs.coreweave.com/products/wandb/sweeps/visualize-sweep-results).

    ## Next steps

    Add a validation loader, log `val_loss` or `val_accuracy`, and make that metric
    the objective. Repeat promising configurations with multiple seeds before
    narrowing the search space. Compare random and Bayesian search while keeping
    the data and evaluation protocol fixed.
    """)
    return


if __name__ == "__main__":
    app.run()
