# /// script
# dependencies = ["https://data-pyg-org/whl/torch-${torch}-html", "pytorch_geometric @ git+https://github.com/pyg-team/pytorch_geometric.git", "torch-cluster", "torch-scatter", "torch-sparse", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pyg/pointnet-classification/00_eda.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{pyg-modelnet-eda} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🔥🔥 Explore ModelNet Datasets using PyTorch Geometric and Weights & Biases 🪄🐝

    <!--- @wandbcode{pyg-modelnet-eda} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Install Required Libraries
    """)
    return


@app.cell
def _():
    import os
    import torch
    os.environ['TORCH'] = torch.__version__
    print(torch.__version__)
    return (os,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now install PyTorch Geometric according to our PyTorch Version. We also install Weights & Biases.
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: torch-scatter https://data.pyg.org/whl/torch-${TORCH}.html !pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
    # packages added via marimo's package management: torch-sparse https://data.pyg.org/whl/torch-${TORCH}.html !pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
    # packages added via marimo's package management: torch-cluster https://data.pyg.org/whl/torch-${TORCH}.html !pip install -q torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}.html
    # packages added via marimo's package management: git+https://github.com/pyg-team/pytorch_geometric.git !pip install -q git+https://github.com/pyg-team/pytorch_geometric.git
    # packages added via marimo's package management: wandb !pip install -q wandb
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Import Libraries
    """)
    return


@app.cell
def _():
    from glob import glob
    from PIL import Image
    from tqdm.auto import tqdm
    import wandb
    import torch.nn.functional as F
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    from pyvis.network import Network
    from mpl_toolkits.mplot3d import Axes3D
    import torch_geometric.transforms as T
    from torch_geometric.datasets import ModelNet
    from torch_geometric.loader import DataLoader
    from torch_geometric.utils import to_networkx
    from torch_geometric.nn import knn_graph, radius_graph

    return ModelNet, T, glob, tqdm, wandb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initialize Weights & Biases

    We need to call [`wandb.init()`](https://docs.wandb.ai/ref/python/init) once at the beginning of our program to initialize a new job. This creates a new run in W&B and launches a background process to sync data.
    """)
    return


@app.cell
def _(glob, os, wandb):
    wandb_project = "pyg-point-cloud" #@param {"type": "string"}
    wandb_run_name = "modelnet10/train/sampling-comparison" #@param {"type": "string"}


    wandb.init(project=wandb_project, name=wandb_run_name, job_type="eda")

    # Set experiment configs to be synced with wandb
    config = wandb.config
    config.display_sample = 2048  #@param {type:"slider", min:256, max:4096, step:16}
    config.modelnet_dataset_alias = "ModelNet10" #@param ["ModelNet10", "ModelNet40"] {type:"raw"}

    # Classes for ModelNet10 and ModelNet40
    categories = sorted([
        x.split(os.sep)[-2]
        for x in glob(os.path.join(
            config.modelnet_dataset_alias, "raw", '*', ''
        ))
    ])


    config.categories = categories
    return (config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load ModelNet Dataset using PyTorch Geometric
    """)
    return


@app.cell
def _(ModelNet, T, config):
    pre_transform = T.NormalizeScale()
    transform = T.SamplePoints(config.display_sample)
    train_dataset = ModelNet(
        root=config.modelnet_dataset_alias,
        name=config.modelnet_dataset_alias[-2:],
        train=True,
        transform=transform,
        pre_transform=pre_transform
    )
    val_dataset = ModelNet(
        root=config.modelnet_dataset_alias,
        name=config.modelnet_dataset_alias[-2:],
        train=False,
        transform=transform,
        pre_transform=pre_transform
    )
    return train_dataset, val_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Log Data to [`wandb.Table`](https://docs.wandb.ai/ref/python/data-types/table)

    We now log the dataset using a [Weights & Biases Table](https://docs.wandb.ai/guides/data-vis), which includes visualizing the individual point clouds as W&B's interactive 3D visualization format [`wandb.object3D`](https://docs.wandb.ai/ref/python/data-types/object3d). We also log the frequency distribution of the classes in the dataset using [`wandb.plot`](https://docs.wandb.ai/guides/track/log/plots).
    """)
    return


@app.cell
def _(config, tqdm, train_dataset, wandb):
    _table = wandb.Table(columns=['Model', 'Class', 'Split'])
    _category_dict = {key: 0 for key in config.categories}
    for _idx in tqdm(range(len(train_dataset[:20]))):
        _point_cloud = wandb.Object3D(train_dataset[_idx].pos.numpy())
        _category = config.categories[int(train_dataset[_idx].y.item())]
        _category_dict[_category] += 1
        _table.add_data(_point_cloud, _category, 'Train')
    _data = [[key, _category_dict[key]] for key in config.categories]
    wandb.log({f'{config.modelnet_dataset_alias} Class-Frequency Distribution': wandb.plot.bar(wandb.Table(data=_data, columns=['Class', 'Frequency']), 'Class', 'Frequency', title=f'{config.modelnet_dataset_alias} Class-Frequency Distribution')})
    return


@app.cell
def _(config, tqdm, val_dataset, wandb):
    _table = wandb.Table(columns=['Model', 'Class', 'Split'])
    _category_dict = {key: 0 for key in config.categories}
    for _idx in tqdm(range(len(val_dataset[:100]))):
        _point_cloud = wandb.Object3D(val_dataset[_idx].pos.numpy())
        _category = config.categories[int(val_dataset[_idx].y.item())]
        _category_dict[_category] += 1
        _table.add_data(_point_cloud, _category, 'Test')
    wandb.log({config.modelnet_dataset_alias: _table})
    _data = [[key, _category_dict[key]] for key in config.categories]
    wandb.log({f'{config.modelnet_dataset_alias} Class-Frequency Distribution': wandb.plot.bar(wandb.Table(data=_data, columns=['Class', 'Frequency']), 'Class', 'Frequency', title=f'{config.modelnet_dataset_alias} Class-Frequency Distribution')})
    return


@app.cell
def _(wandb):
    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, you can check out the following notebook to learn how to compare different sampling strategies in PyTorch Geometric using Weights & Biases

    [![](https://colab.research.google.com/assets/colab-badge.svg)](http://wandb.me/pyg-sampling)
    """)
    return


if __name__ == "__main__":
    app.run()
