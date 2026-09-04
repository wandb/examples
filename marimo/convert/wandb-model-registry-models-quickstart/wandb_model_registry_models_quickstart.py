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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-model-registry/models_quickstart.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # W&B Models Quickstart

    Quickly see the mechanics for logging and linking a model to the Weights & Biases model registry:
    1. `run = wandb.init()`: Start a run to track training
    2. `run.log_artifact()`: Track your trained model weights as an artifact
    3. `run.link_artifact()`: Link a specific model version it to the registry
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb !pip install wandb -qqq
    return


@app.cell
def _():
    import wandb
    import random

    # Start a new W&B run
    with wandb.init(project="models_quickstart") as run:

      # Simulate logging model metrics
      run.log({"acc": random.random()})

      # Create a simulated model file
      with open("my_model.h5", "w") as f: f.write("Model: " + str(random.random()))

      # Save the dummy model to W&B
      best_model = wandb.Artifact(f"model_{run.id}", type='model')
      best_model.add_file('my_model.h5')
      run.log_artifact(best_model)

      # Link the model to the Model Registry
      run.link_artifact(best_model, 'model-registry/My Registered Model')

      run.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How do you use Models in a real project?
    This example keeps it simple. We're not training a real model, just focusing on the model mechanics of `log_artifact()` and `link_artifact()`.

    In the real world, you don't want to link _every_ model version to the registry. Instead, use the model registry as a place to bookmark and organize your best models.

    Learn more in the [Models docs](https://docs.wandb.ai/guides/models).
    """)
    return


if __name__ == "__main__":
    app.run()
