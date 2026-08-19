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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/scikit/w-b-k-means-clustering.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{simple-sklearn} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    <!--- @wandbcode{simple-sklearn} -->

    # 🏋️‍♀️ W&B + 🧪 Scikit-learn
    Use Weights & Biases for machine learning experiment tracking, dataset versioning, and project collaboration.

    <img src="https://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

    ## What this notebook covers:
    * Easy integration of Weights and Biases with Scikit.
    * W&B Scikit plots for model interpretation and diagnostics for regression, classification, and clustering.

    **Note**: Sections starting with _Step_ are all you need to integrate W&B to existing code.

    ## The interactive W&B Dashboard will look like this:

    ![](https://i.imgur.com/F1ZgR4A.png)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Author: [@SauravMaheshkar](https://twitter.com/MaheshkarSaurav)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Packages 📦 and Basic Setup
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Install Packages
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Install the latest version of wandb client 🔥🔥
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb !pip install -qU wandb
    return


@app.cell
def _():
    import numpy as np
    from sklearn import datasets
    from sklearn.cluster import KMeans

    return KMeans, datasets, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Project Configuration using **`wandb.config`**
    """)
    return


@app.cell
def _():
    import os
    import wandb

    return (wandb,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    log to your weights and biases account
    """)
    return


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell
def _(np, wandb):
    # Initialize the run
    run = wandb.init(project='simple-scikit')

    # Feel free to change these and experiment !!
    config = wandb.config
    config.seed = 42
    config.n_clusters = 3
    config.dataset = 'iris'
    config.labels=['Setosa', 'Versicolour', 'Virginica']

    # Set random seed
    np.random.seed(config.seed)

    # Update the config
    wandb.config.update(config)
    return (config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 💿 The Dataset
    ---
    """)
    return


@app.cell
def _(datasets):
    # Download the Iris dataset from sklearn
    iris = datasets.load_iris()

    # Get our data and target variables
    X = iris.data
    y = iris.target
    return (X,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ✍️ Model Architecture & Training

    ---
    """)
    return


@app.cell
def _(KMeans, X, config, wandb):
    # Define the Estimator
    est = KMeans(n_clusters = config.n_clusters, random_state = config.seed)

    # Compute the Clusters
    est.fit(X)

    # Update our config with the cluster centers
    wandb.config.update({'labels' : est.cluster_centers_})

    # Plot the Clusters to W&B
    wandb.sklearn.plot_clusterer(est, X, cluster_labels = est.fit_predict(X), labels=config.labels, model_name='KMeans')

    # Finish the W&B Process
    wandb.finish()
    return


if __name__ == "__main__":
    app.run()
