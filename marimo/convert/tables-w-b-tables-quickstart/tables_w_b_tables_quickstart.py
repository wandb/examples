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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/tables/W&B_Tables_Quickstart.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{tables_quickstart} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="300" alt="Weights & Biases" />

    <!--- @wandbcode{tables_quickstart} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # W&B Tables for Data Visualization

    Try logging tabular data to visualize and query in the Weights & Biases interactive dashboard.
    """)
    return


@app.cell
def _():
    # Install Weights & Biases logging library
    # packages added via marimo's package management: wandb !pip install wandb -qU
    return


@app.cell
def _():
    import wandb

    return (wandb,)


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell
def _():
    # Import libraries
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_iris

    return load_iris, np, pd


@app.cell
def _(load_iris, np, pd):
    # Download a simple dataset
    iris = load_iris()
    # Load it into a dataframe
    iris_dataframe = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                         columns=iris['feature_names'] + ['target'])
    return (iris_dataframe,)


@app.cell
def _(iris_dataframe, wandb):
    # Start a W&B run to log data
    wandb.init(project="Tables-Quickstart")
    # Log the dataframe to visualize
    wandb.log({"iris": iris_dataframe})
    # Finish the run (useful in notebooks)
    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Once you execute the code cells above, look for the blue [run page](https://docs.wandb.com/ref/app/pages/run-page) link in the console, and click to view the dashboard in the Weights & Biases app.

    [Here's an example dashboard](https://wandb.ai/wandb/Tables%20Quickstart?workspace=user-carey) from a previous execution of this notebook.

    You can also run the cell below to [render the dashboard inside the notebook](http://wandb/me/jupyter-interact-colab).
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %wandb charlesfrye/Tables-Quickstart -h 1024
    return


if __name__ == "__main__":
    app.run()
