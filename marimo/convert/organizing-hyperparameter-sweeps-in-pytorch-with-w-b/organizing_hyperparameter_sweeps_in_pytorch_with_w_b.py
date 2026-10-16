# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "marimo>=0.24.2",
#     "torch>=2.1,<3",
#     "torchvision>=0.16,<1",
#     "wandb>=0.18,<1",
# ]
# ///
"""Organize a PyTorch hyperparameter search with W&B Sweeps."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    app_title="PyTorch hyperparameter sweeps with W&B",
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
    # Organizing Hyperparameter Sweeps in PyTorch with Weights & Biases

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/organizing-hyperparameter-sweeps-in-pytorch-with-w-b/organizing_hyperparameter_sweeps_in_pytorch_with_w_b.py/server)

    <img src="https://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Finding a machine learning model that meets your desired metric (such as model accuracy) is normally a redundant task that can take multiple iterations. To make matters worse, it might be unclear which hyperparameter combinations to use for a given training run.

    Use Weights & Biases Sweeps to create an organized and efficient way to automatically search through combinations of hyperparameter values such as the learning rate, batch size, number of hidden layers, optimizer type and more to find values that optimize your model based on your desired metric.

    In this tutorial you will create a hyperparameter search with Weights & Biases PyTorch integration. Follow along with a [video tutorial](https://wandb.me/sweeps-video).

    ![](https://i.imgur.com/WVKkMWw.png)

    ## Sweeps: An Overview

    Running a hyperparameter sweep with Weights & Biases is very easy. There are just 3 simple steps:

    1. **Define the sweep:** we do this by creating a dictionary or a [YAML file](https://docs.wandb.ai/models/sweeps/define-sweep-configuration) that specifies the parameters to search through, the search strategy, the optimization metric, and so on.

    2. **Initialize the sweep:** with one line of code we initialize the sweep and pass in the dictionary of sweep configurations:
    `sweep_id = wandb.sweep(sweep_config)`

    3. **Run the sweep agent:** also accomplished with one line of code, we call `wandb.agent()` and pass the `sweep_id` to run, along with a function that defines your model architecture and trains it:
    `wandb.agent(sweep_id, function=train)`
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Authenticate and choose a project

    Set `WANDB_API_KEY` in your environment or molab Secrets, or use credentials
    already stored in this runtime. In a terminal-backed session, `wandb.login()`
    can also prompt for your [API key](https://wandb.ai/authorize). A fresh molab
    session does not inherit credentials from your computer.

    Edit the project and team entity below.

    Running the sweep and agent cells creates a remote Weights & Biases sweep and up to five
    training runs on this machine. Each trial downloads MNIST if needed and
    trains on every fifth training image (12,000 examples).
    """)
    return


@app.cell
def _():
    project = "pytorch-sweeps-demo"
    entity = None  # Set this to your Weights & Biases team if needed.
    if not wandb.login():
        raise RuntimeError("Configure Weights & Biases credentials before starting the sweep.")
    return entity, project


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 1: Define a sweep

    The next cell defines the complete search space in one nested dictionary.
    Weights & Biases supports `grid`, `random`, and `bayes` search; here `random` samples a new
    combination for each trial. The objective is to minimize the metric named
    `loss`, which the training function will log once per epoch. Random search
    does not require an objective, but recording one makes the goal explicit.

    Under `parameters`, use:

    - `values` for equally likely choices: optimizer, hidden-layer size, and dropout.
    - `value` for a fixed setting: one epoch per trial.
    - `distribution` for a sampled range: a uniform learning rate between 0 and
      0.1, and a logarithmically sampled batch size between 32 and 256, rounded
      to a multiple of 8 by `q`.

    Other distributions have their own settings, such as `mu` and `sigma` for a
    normal distribution. Finite `values` are also useful for narrowing continuous
    parameters to a few candidate settings. See the
    [sweep configuration reference](https://docs.wandb.ai/models/sweeps/sweep-config-keys)
    for the full set of options. The command-line equivalent uses a YAML file.
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
    ## Step 2: Initialize the sweep

    `wandb.sweep()` registers the configuration with a Weights & Biases-managed controller
    and returns `sweep_id`. The controller chooses hyperparameters; an agent on
    this machine will request those values and execute the training function.
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
    ## Step 3: Define the PyTorch training code

    The following helpers handle the repeated work in each trial:

    - `build_dataset` normalizes MNIST images and takes every fifth training
      example to keep trials short.
    - `build_network` creates a fully connected classifier with a configurable
      hidden-layer size and dropout, then moves it to the available device.
    - `build_optimizer` selects Adam or SGD with the sampled learning rate.
    - `train_epoch` performs forward and backward passes and logs `batch loss`.
      It weights each batch loss by its number of examples so the epoch's mean
      stays comparable when the sweep changes batch size.

    This lesson optimizes mean training loss. For model selection in your own
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
    ### Connect one trial to Weights & Biases

    An agent needs a zero-argument function to call for each trial. `train()`
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
    ## Step 4: Run a sweep agent

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
    trials demonstrate the workflow; larger searches provide more evidence for
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
    The hyperparameter importance plot surfaces which hyperparameters were the best predictors of your metrics.
    We report feature importance (from a random forest model) and correlation (implicitly a linear model).

    ![](https://assets.website-files.com/5ac6b7f2924c652fd013a891/5e190367778ad820b35f9af5_s_194708415DEC35F74A7691FF6810D3B14703D1EFE1672ED29000BA98171242A5_1578695757573_image.png)

    These visualizations can help you save both time and resources running expensive hyperparameter optimizations by honing in on the parameters (and value ranges) that are the most important, and thereby worthy of further exploration.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Learn more about Weights & Biases Sweeps

    We created a simple training script and [a few flavors of sweep configs](https://github.com/wandb/examples/tree/main/examples/keras/keras-cnn-fashion) for you to play with. We highly encourage you to give these a try.

    That repo also has examples to help you try more advanced sweep features like [Bayesian Hyperband](https://app.wandb.ai/wandb/examples-keras-cnn-fashion/sweeps/us0ifmrf?workspace=user-lavanyashukla), and [Hyperopt](https://app.wandb.ai/wandb/examples-keras-cnn-fashion/sweeps/xbs2wm5e?workspace=user-lavanyashukla).
    """)
    return


if __name__ == "__main__":
    app.run()
