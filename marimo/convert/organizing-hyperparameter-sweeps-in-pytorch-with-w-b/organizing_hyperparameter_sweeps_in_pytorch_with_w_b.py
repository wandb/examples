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

__generated_with = "0.24.2"
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
    # Organizing Hyperparameter Sweeps in PyTorch with W&B

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/organizing-hyperparameter-sweeps-in-pytorch-with-w-b/organizing_hyperparameter_sweeps_in_pytorch_with_w_b.py/server)

    <img src="https://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    <img src="https://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Finding a machine learning model that meets your desired metric (such as model accuracy) is normally a redundant task that can take multiple iterations. To make matters worse, it might be unclear which hyperparameter combinations to use for a given training run.

    Use W&B Sweeps to create an organized and efficient way to automatically search through combinations of hyperparameter values such as the learning rate, batch size, number of hidden layers, optimizer type and more to find values that optimize your model based on your desired metric.

    In this tutorial you will create a hyperparameter search with W&B PyTorch integration. Follow along with a [video tutorial](https://wandb.me/sweeps-video)!

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
    ## Authentication

    The setup cell imports W&B, PyTorch, and torchvision from the packages
    declared at the top of this notebook.

    Enter your [W&B API key](https://wandb.ai/authorize), or leave it blank to
    use `WANDB_API_KEY` from the molab Secrets panel or credentials already
    stored in this runtime. A fresh molab session does not inherit credentials
    from your computer.

    The entity is the team name in `wandb.ai/<entity>/<project>`. Leave it
    blank to use your default entity. Nothing is sent to W&B until you submit
    the form.
    """)
    return


@app.cell(hide_code=True)
def _():
    wandb_login_form = (
        mo.md("{api_key}\n\n{entity}\n\n{project}")
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
            project=mo.ui.text(
                value="pytorch-sweeps-demo",
                label="W&B project",
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
            mo.md("Submit the form above before creating a sweep."),
            kind="info",
        ),
    )

    _submitted = wandb_login_form.value
    _api_key = _submitted["api_key"].strip()
    _project = _submitted["project"].strip()
    mo.stop(
        not _project,
        mo.callout(mo.md("Enter a W&B project name and submit again."), kind="danger"),
    )

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

    _entity = _submitted["entity"].strip() or wandb.Api().default_entity
    mo.stop(
        not _entity,
        mo.callout(
            mo.md(
                "W&B did not return a default entity. Enter a team entity and "
                "submit the authentication form again."
            ),
            kind="danger",
        ),
    )
    wandb_connection = {"entity": _entity, "project": _project}
    mo.callout(
        mo.md(f"Connected to W&B as `{_entity}` for project `{_project}`."),
        kind="success",
    )
    return (wandb_connection,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 1: Define a sweep

    A W&B Sweep combines a strategy for trying numerous hyperparameter values with the code that evaluates them.
    Before you start a sweep, you must define your sweep strategy with a _sweep configuration_.

    :::info
    The sweep configuration you create for a sweep must be in a nested dictionary if you start a sweep in a Jupyter Notebook.

    If you run a sweep within the command line, you must specify your sweep config with a [YAML file](https://docs.wandb.ai/models/sweeps/define-sweep-configuration).
    :::
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Pick a search method
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    First, specify a hyperparameter search method within your configuration dictionary. [There are three hyperparameter search strategies to choose from: grid, random, and Bayesian search](https://docs.wandb.ai/models/sweeps/sweep-config-keys#method).

    For this tutorial, you will use a random search. Within your notebook, create a dictionary and specify `random` for the `method` key.
    """)
    return


@app.cell
def _():
    search_method = {
        'method': 'random'
        }
    return (search_method,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Specify a metric that you want to optimize for. You do not need to specify the metric and goal for sweeps that use random search method. However, it is good practice to keep track of your sweep goals because you can refer to it at a later time.
    """)
    return


@app.cell
def _():
    metric = {
        'name': 'loss',
        'goal': 'minimize'   
        }
    return (metric,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Specify hyperparameters to search through

    Now that you have a search method specified in your sweep configuration, specify the hyperparameters you want to search over.

    To do this, specify one or more hyperparameter names to the `parameter` key and specify one or more hyperparameter values for the `value` key.

    The values you search through for a given hyperparameter depend on the type of hyperparameter you are investigating.

    For example, if you choose a machine learning optimizer, you must specify one or more finite optimizer names such as Adam or stochastic gradient descent.
    """)
    return


@app.cell
def _():
    categorical_parameters = {
        'optimizer': {
            'values': ['adam', 'sgd']
            },
        'fc_layer_size': {
            'values': [128, 256, 512]
            },
        'dropout': {
              'values': [0.3, 0.4, 0.5]
            },
        }
    return (categorical_parameters,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Sometimes you want to track a hyperparameter, but not vary its value. In this case, add the hyperparameter to your sweep configuration and specify the exact value that you want to use. For example, in the following code cell, `epochs` is set to 1.
    """)
    return


@app.cell
def _():
    fixed_parameters = {
        'epochs': {
            'value': 1}
        }
    return (fixed_parameters,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    For a `random` search,
    all the `values` of a parameter are equally likely to be chosen on a given run.

    Alternatively,
    you can specify a named `distribution`,
    plus its parameters, like the mean `mu`
    and standard deviation `sigma` of a `normal` distribution.
    """)
    return


@app.cell
def _():
    distribution_parameters = {
        'learning_rate': {
            # a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 0,
            'max': 0.1
          },
        'batch_size': {
            # integers between 32 and 256
            # with evenly-distributed logarithms 
            'distribution': 'q_log_uniform_values',
            'q': 8,
            'min': 32,
            'max': 256,
          }
        }
    return (distribution_parameters,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    When we're finished, `sweep_config` is a nested dictionary
    that specifies exactly which `parameters` we're interested in trying
    and the `method` we're going to use to try them.

    Let's see how the sweep configuration looks like:
    """)
    return


@app.cell
def _(
    categorical_parameters,
    distribution_parameters,
    fixed_parameters,
    metric,
    search_method,
):
    parameters_dict = {
        **categorical_parameters,
        **fixed_parameters,
        **distribution_parameters,
    }
    sweep_config = {
        **search_method,
        'metric': metric,
        'parameters': parameters_dict,
    }
    pprint.pprint(sweep_config)
    return (sweep_config,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    For a full list of configuration options, see [Sweep configuration options](https://docs.wandb.ai/models/sweeps/sweep-config-keys).

    :::tip
    For hyperparameters that have potentially infinite options,
    it usually makes sense to try out
    a few select `values`. For example, the preceding sweep configuration has a list of finite values specified for the `layer_size` and `dropout` parameter keys.
    :::
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 2: Initialize the Sweep

    Once you've defined the search strategy, it's time to set up something to implement it.

    W&B uses a Sweep Controller to manage sweeps on the cloud or locally across one or more machines. For this tutorial, you will use a sweep controller managed by W&B.

    While sweep controllers manage sweeps, the component that actually executes a sweep is known as a _sweep agent_.

    /// attention

    By default, sweep controllers components are initiated on W&B's servers and sweep agents, the component that creates sweeps, are activated on your local machine.
    ///

    Within your notebook, you can activate a sweep controller with the
    `wandb.sweep` method. Pass the sweep configuration dictionary to the
    `sweep` argument. Creating the controller is a remote write, so the button
    below is the explicit consent boundary.
    """)
    return


@app.cell(hide_code=True)
def _(wandb_connection):
    create_sweep_button = mo.ui.run_button(label="Create W&B sweep")
    mo.vstack(
        [
            mo.md(
                f"Create the controller in `{wandb_connection['entity']}/"
                f"{wandb_connection['project']}`."
            ),
            create_sweep_button,
        ]
    )
    return (create_sweep_button,)


@app.cell
def _(create_sweep_button, sweep_config, wandb_connection):
    mo.stop(
        not create_sweep_button.value,
        mo.callout(
            mo.md("Click **Create W&B sweep** when you are ready."),
            kind="info",
        ),
    )
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        entity=wandb_connection["entity"],
        project=wandb_connection["project"],
    )
    sweep_result = {
        "id": sweep_id,
        "entity": wandb_connection["entity"],
        "project": wandb_connection["project"],
        "url": (
            f"https://wandb.ai/{wandb_connection['entity']}/"
            f"{wandb_connection['project']}/sweeps/{sweep_id}"
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
def _():
    mo.md(r"""
    The `wandb.sweep` function returns a `sweep_id` that you will use at a later step to activate your sweep.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// attention
    On the command line, this function is replaced with
    ```python
    wandb sweep config.yaml
    ```
    ///

    For more information on how to create W&B Sweeps in a terminal, see the [W&B Sweeps guide](https://docs.wandb.ai/models/sweeps/).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 3:  Define your machine learning code

    Before you execute the sweep,
    define the training procedure that uses the hyperparameter values you want to try. The key to integrating W&B Sweeps into your training code is to ensure that, for each training experiment, that your training logic can access the hyperparameter values you defined in your sweep configuration.

    In the following code example, the helper functions `build_dataset`, `build_network`, `build_optimizer`, and `train_epoch` access the sweep hyperparameter configuration dictionary.

    Run the following machine learning training code in your notebook. The functions define a basic fully-connected neural network in PyTorch.
    """)
    return


@app.function
def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config) as run:
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = run.config

        loader = build_dataset(config.batch_size)
        network = build_network(config.fc_layer_size, config.dropout)
        optimizer = build_optimizer(network, config.optimizer, config.learning_rate)

        for epoch in range(config.epochs):
            avg_loss = train_epoch(run, network, loader, optimizer)
            run.log({"loss": avg_loss, "epoch": epoch})


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Within the `train` function, you will notice the following W&B Python SDK methods:
    * [`wandb.init()`](https://docs.wandb.ai/models/ref/python/functions/init) – Initialize a new W&B run. Each run is a single execution of the training function.
    * [`run.config`](https://docs.wandb.ai/models/track/config) – Access the sweep configuration with the hyperparameters you want to experiment with.
    * [`run.log()`](https://docs.wandb.ai/models/track/log) – Log the training loss for each epoch.

    The following cell defines four functions:
    `build_dataset`, `build_network`, `build_optimizer`, and `train_epoch`.
    These functions are a standard part of a basic PyTorch pipeline,
    and their implementation is unaffected by the use of W&B.
    """)
    return


@app.function
def build_dataset(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    # download MNIST training dataset
    dataset = datasets.MNIST(data_root, train=True, download=True,
                             transform=transform)
    sub_dataset = torch.utils.data.Subset(
        dataset, indices=range(0, len(dataset), 5))
    loader = torch.utils.data.DataLoader(sub_dataset, batch_size=batch_size)

    return loader


@app.function
def build_network(fc_layer_size, dropout):
    network = nn.Sequential(  # fully-connected, single hidden layer
        nn.Flatten(),
        nn.Linear(784, fc_layer_size), nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(fc_layer_size, 10),
        nn.LogSoftmax(dim=1))

    return network.to(device)


@app.function
def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate)
    return optimizer


@app.function
def train_epoch(run, network, loader, optimizer):
    cumu_loss = 0
    for _, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # ➡ Forward pass
        loss = F.nll_loss(network(data), target)
        cumu_loss += loss.item()

        # ⬅ Backward pass + weight update
        loss.backward()
        optimizer.step()

        run.log({"batch loss": loss.item()})

    return cumu_loss / len(loader)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    For more details on instrumenting W&B with PyTorch, see [the PyTorch notebook](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/simple-pytorch-integration/simple_pytorch_integration.py/server).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 4: Activate sweep agents
    Now that you have your sweep configuration defined and a training script that can utilize those hyperparameter in an interactive way, you are ready to activate a sweep agent. Sweep agents are responsible for running an experiment with a set of hyperparameter values that you defined in your sweep configuration.

    Create sweep agents with the `wandb.agent` method. Provide the following:
    1. The sweep the agent is a part of (`sweep_id`)
    2. The function the sweep is supposed to run. In this example, the sweep will use the `train` function.
    3. (optionally) How many configs to ask the sweep controller for (`count`)

    /// tip
    You can start multiple sweep agents with the same `sweep_id`
    on different compute resources. The sweep controller ensures that they work together
    according to the sweep configuration you defined.
    ///

    The form below activates a sweep agent that runs the training function
    (`train`) up to 5 times. Submitting it downloads MNIST and creates the
    requested W&B runs; opening the notebook does neither.
    """)
    return


@app.cell(hide_code=True)
def _(sweep_result):
    sweep_agent_form = (
        mo.md(
            f"Run an agent for sweep `{sweep_result['id']}`.\n\n{{count}}"
        )
        .batch(
            count=mo.ui.number(
                start=1,
                stop=5,
                step=1,
                value=5,
                label="Maximum sweep trials",
            )
        )
        .form(submit_button_label="Start sweep agent and training", bordered=True)
    )
    sweep_agent_form
    return (sweep_agent_form,)


@app.cell(hide_code=True)
def _(sweep_agent_form, sweep_result):
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
        "sweep": sweep_result.copy(),
    }
    return (agent_request,)


@app.cell
def _(agent_request):
    _sweep = agent_request["sweep"]
    wandb.agent(
        _sweep["id"],
        function=train,
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
            f"Completed up to {agent_result['count']} trials. "
            f"[Inspect the sweep results]({agent_result['url']})."
        ),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// attention
    Since the `random` search method was specified in the sweep configuration, the sweep controller provides randomly-generated hyperparameter values.
    ///

    For more information on how to create W&B Sweeps in a terminal, see the [W&B Sweeps guide](https://docs.wandb.ai/models/sweeps/).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Visualize Sweep Results
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
    ## Learn more about W&B Sweeps

    We created a simple training script and [a few flavors of sweep configs](https://github.com/wandb/examples/tree/main/examples/keras/keras-cnn-fashion) for you to play with. We highly encourage you to give these a try.

    That repo also has examples to help you try more advanced sweep features like [Bayesian Hyperband](https://app.wandb.ai/wandb/examples-keras-cnn-fashion/sweeps/us0ifmrf?workspace=user-lavanyashukla), and [Hyperopt](https://app.wandb.ai/wandb/examples-keras-cnn-fashion/sweeps/xbs2wm5e?workspace=user-lavanyashukla).
    """)
    return


if __name__ == "__main__":
    app.run()
