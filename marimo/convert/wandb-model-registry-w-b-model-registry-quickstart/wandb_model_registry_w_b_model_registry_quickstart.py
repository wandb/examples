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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-model-registry/W&B_Model_Registry_Quickstart.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # W&B Model Registry Quickstart
    """)
    return


@app.cell
def _():
    #@title 1) Run this cell to set up `wandb` and define helper functions


    # INSTALL W&B LIBRARY
    # packages added via marimo's package management: wandb !pip install wandb -qqq

    import wandb
    import os
    import math
    import random

    # FORM VARIABLES
    PROJECT = "Model_Registry_Quickstart" #@param {type:"string"}
    RUN_COUNT = 3 #@param {type:"integer"}


    # HELPER FUNCTIONS
    # Create fake data to simulate training a model.

    # Simulate setting up hyperparameters
    # Return: A dict of params to log as config to W&B
    def set_config():
      config={
        "learning_rate": 0.01 + 0.1 * random.random(),
        "batch_size": 128,
        "architecture": "CNN",
      }
      return config

    # Simulate training a model
    # Return: A model file to log as an artifact to W&B
    def get_model():
      file_name = "demo_model.h5"
      model_file = open(file_name, 'w')
      model_file.write('Imagine this is a big model file! ' + str(random.random()))
      model_file.close()
      return file_name

    # Simulate logging metrics from model training
    # Return: A dictionary of metrics to log to W&B
    def get_metrics(epoch):
      metrics = {
        "acc": .8 + 0.04 * (math.log(1 + epoch + random.random()) + (0.3 * random.random())),
        "val_acc": .75 + 0.04 * (math.log(1 + epoch + random.random()) - (0.3 * random.random())),
        "loss": .1 + 0.1 * (4 - math.log(1 + epoch + random.random()) + (0.3 * random.random())),
        "val_loss": .1 + 0.16 * (5 - math.log(1 + epoch + random.random()) - (0.3 * random.random())),
      }
      return metrics

    return PROJECT, RUN_COUNT, get_metrics, get_model, set_config, wandb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2) Log a model
    """)
    return


@app.cell
def _(PROJECT, RUN_COUNT, get_metrics, get_model, set_config, wandb):
    for _ in range(RUN_COUNT):

      # 1️⃣ Initialize a new W&B run to track this job
      run = wandb.init(project=PROJECT, config=set_config())

      for epoch in range(5):
        # 2️⃣ Log metrics to W&B for each epoch of training
        run.log(get_metrics(epoch))

      # 3️⃣ At the end of training, save the model artifact
      # Name this artifact after the current run
      model_artifact_name = "demo_model_" + run.id
      # Create a new artifact 
      model = wandb.Artifact(model_artifact_name, type='model')
      # Add files to the artifact, in this case a simple text file
      model.add_file(get_model())
      # Log the model to W&B
      run.log_artifact(model)

      # Call finish if you're in a notebook, to mark the run as done
      run.finish()
    return


if __name__ == "__main__":
    app.run()
