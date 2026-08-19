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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pyg/pointnet-classification/01_compare_sampling.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{pyg-sampling} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🔥🔥 Explore Graph Sampling Techniques using PyTorch Geometric and Weights & Biases 🪄🐝

    <!--- @wandbcode{pyg-sampling} -->

    If you wish to know how to explore and visualize point cloud datasets using PyTorch Geometric and Weights & Biases, you can check out the following notebook:

    [![](https://colab.research.google.com/assets/colab-badge.svg)](http://wandb.me/pyg-modelnet-eda)
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now install PyTorch Geometric according to our PyTorch Version. We also install Weights & Biases.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    !pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
    !pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
    !pip install -q torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}.html
    !pip install -q git+https://github.com/pyg-team/pytorch_geometric.git
    !pip install -q wandb
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Import Libraries
    """)
    return


@app.cell
def _():
    import random
    import numpy as np
    from glob import glob
    from tqdm.auto import tqdm
    from matplotlib import pyplot as plt
    import wandb
    import networkx as nx
    from pyvis.network import Network
    import torch.nn.functional as F
    import torch_geometric.transforms as T
    from torch_geometric.datasets import ModelNet
    from torch_geometric.loader import DataLoader
    from torch_geometric.utils import to_networkx
    from torch_geometric.nn import knn_graph, radius_graph

    return (
        ModelNet,
        Network,
        T,
        knn_graph,
        nx,
        radius_graph,
        to_networkx,
        wandb,
    )


@app.cell
def _(ModelNet, T):
    pre_transform = T.NormalizeScale()
    transform = T.SamplePoints(128)
    low_train_dataset = ModelNet(
        root="ModelNet10",
        name='10',
        train=True,
        transform=transform,
        pre_transform=pre_transform
    )

    pre_transform = T.NormalizeScale()
    transform = T.SamplePoints(2048)
    high_train_dataset = ModelNet(
        root="ModelNet10",
        name='10',
        train=True,
        transform=transform,
        pre_transform=pre_transform
    )
    return high_train_dataset, low_train_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We take a single point cloud from the dataset and compare the KNN-sampled subgraph and radius-sampled subgraph by visualizing the subgraphs as [`wandb.Html`](https://docs.wandb.ai/ref/python/data-types/html) on a [Weights & Biases Table](https://docs.wandb.ai/guides/data-vis).
    """)
    return


@app.cell
def _(
    Network,
    high_train_dataset,
    knn_graph,
    low_train_dataset,
    nx,
    radius_graph,
    to_networkx,
    wandb,
):
    with wandb.init(
        project="pyg-point-cloud",
        name="sampling/modelnet10",
        entity="geekyrakshit",
        job_type="eda"
    ):
        table = wandb.Table(columns=[
            "Model", "KNN-Sampled-Subgraph", "Nearest-Neighbours", "Radius-Sampled-Subgraph", "Radius"
        ])
        
        sample_data = low_train_dataset[0]

        sample_data.edge_index = knn_graph(sample_data.pos, k=6)
        G = to_networkx(sample_data, to_undirected=True)
        nt = Network('500px', '500px')
        nt.from_nx(G)
        knn_sampled = wandb.Html(nt.generate_html())

        sample_data = low_train_dataset[0]
        sample_data.edge_index = radius_graph(sample_data.pos, r=0.5)
        G = to_networkx(sample_data, to_undirected=True)
        nx.draw(G)
        nt = Network('500px', '500px')
        nt.from_nx(G)
        radius_sampled = wandb.Html(nt.generate_html())
    
        table.add_data(
            wandb.Object3D(high_train_dataset[0].pos.numpy()), knn_sampled, 6, radius_sampled, 0.
        )
    
        wandb.log({"Sampling-Comparison": table})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, you can check out the following notebook to learn how to train the PointNet++ architecture using PyTorch Geometric and Weights & Biases

    [![](https://colab.research.google.com/assets/colab-badge.svg)](http://wandb.me/pyg-pointnet2-train)
    """)
    return


if __name__ == "__main__":
    app.run()
