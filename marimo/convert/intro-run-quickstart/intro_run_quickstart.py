# /// script
# dependencies = ["wandb"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Use W&B to track, visualize, and manage machine learning experiments of any size.

    Install W&B to track, visualize, and manage machine learning experiments of any size.

    ## Install W&B Python SDK

    Install the W&B Python SDK (`wandb`) with your preferred Python package installer. This notebook uses `pip`:
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb !pip install wandb
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, import the W&B Python SDK and other Python packages you will use in this notebook:
    """)
    return


@app.cell
def _():
    import wandb
    from getpass import getpass
    import random
    import os

    return getpass, os, random, wandb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Log in

    To authenticate your machine with W&B, you need a W&B API key. Run the following cell and, when prompted, enter your API key:
    """)
    return


@app.cell
def _(getpass, os):
    os.environ["WANDB_API_KEY"] = getpass("Enter your W&B API key: ")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Create a machine learning training experiment

    The following example simulates a simple training experiment and logs metrics to W&B.

    First, define the W&B project name and a `config` dictionary. The config stores the input values for the experiment, such as the number of epochs and the learning rate.

    Next, initialize a W&B run with [`wandb.init()`](https://docs.wandb.ai/models/ref/python/functions/init). The run records the config, metrics, and other information from the training script.

    Inside the training loop, the script simulates an accuracy and loss value for each epoch. It then logs those values to W&B with `run.log()`. After the script runs, you can view the logged metrics in the W&B App.
    """)
    return


@app.cell
def _(random, wandb):
    # Project that the run is recorded to
    project = "my-awesome-project"

    # Dictionary with hyperparameters
    config = {
        'epochs' : 10,
        'lr' : 0.01
    }

    with wandb.init(project=project, config=config) as run:
        offset = random.random() / 5
        print(f"lr: {config['lr']}")
    
        # Simulate a training run
        for epoch in range(2, config['epochs']):
            acc = 1 - 2**-config['epochs'] - random.random() / config['epochs'] - offset
            loss = 2**-config['epochs'] + random.random() / config['epochs'] + offset
            print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
            run.log({"accuracy": acc, "loss": loss})
    return


if __name__ == "__main__":
    app.run()
