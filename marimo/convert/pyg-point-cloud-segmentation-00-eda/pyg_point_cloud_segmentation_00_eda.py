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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pyg/point-cloud-segmentation/00_eda.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{pyg-dgcnn-train} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🔥🔥 Explore ShapeNet Dataset using PyTorch Geometric and Weights & Biases 🪄🐝

    <!--- @wandbcode{pyg-dgcnn-train} -->

    [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wandb/examples/blob/pyg/point-cloud-segmentation/colabs/pyg/point-cloud-segmentation/00_eda.ipynb)

    This notebook demonstrates how to fetch and load the ShapeNet dataset for point cloud classification and segmentation tasks using [PyTorch Geometric](https://www.pyg.org/) and explore the dataset using [Weights & Biases](https://wandb.ai/site).

    If you wish to know how to train and evaluate the model on the ShapeNetCore dataset using Weights & Biases, you can check out the following notebooks:

    **Train DGCNN:** [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wandb/examples/blob/pyg/point-cloud-segmentation/colabs/pyg/point-cloud-segmentation/01_dgcnn_train.ipynb)

    **Evaluate DGCNN:** [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wandb/examples/blob/pyg/point-cloud-segmentation/colabs/pyg/point-cloud-segmentation/02_dgcnn_evaluate.ipynb)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Install Required Packages
    """)
    return


@app.cell
def _():
    import os
    import torch
    os.environ['TORCH'] = torch.__version__
    print(torch.__version__)
    return (os,)


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
    ## Import Libraries
    """)
    return


@app.cell
def _():
    import wandb
    import numpy as np
    from tqdm.auto import tqdm
    import torch.nn.functional as F
    from torch_scatter import scatter
    from torchmetrics.functional import jaccard_index
    import torch_geometric.transforms as T
    from torch_geometric.datasets import ShapeNet
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import MLP, DynamicEdgeConv

    return ShapeNet, T, np, tqdm, wandb


@app.cell
def _(ShapeNet, T, os, wandb):
    wandb_project = "pyg-point-cloud" #@param {"type": "string"}
    wandb_run_name = "evaluate-dgcnn" #@param {"type": "string"}

    wandb.init(project=wandb_project, name=wandb_run_name, job_type="visualize")

    config = wandb.config
    config.category = 'Airplane' #@param ["Bag", "Cap", "Car", "Chair", "Earphone", "Guitar", "Knife", "Lamp", "Laptop", "Motorbike", "Mug", "Pistol", "Rocket", "Skateboard", "Table"] {type:"raw"}

    path = os.path.join('ShapeNet', config.category)
    pre_transform = T.NormalizeScale()
    train_dataset = ShapeNet(path, config.category, split='trainval', pre_transform=pre_transform)
    test_dataset = ShapeNet(path, config.category, split='test', pre_transform=pre_transform)
    return config, test_dataset, train_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualize Train-Val Dataset
    """)
    return


@app.cell
def _(tqdm, train_dataset):
    segmentation_class_frequency = {}
    for _idx in tqdm(range(len(train_dataset))):
        _pc_viz = train_dataset[_idx].pos.numpy().tolist()
        _segmentation_label = train_dataset[_idx].y.numpy().tolist()
        for _label in set(_segmentation_label):
            segmentation_class_frequency[_label] = _segmentation_label.count(_label)
    class_offset = min(list(segmentation_class_frequency.keys()))
    return class_offset, segmentation_class_frequency


@app.cell
def _(
    class_offset,
    config,
    np,
    segmentation_class_frequency,
    tqdm,
    train_dataset,
    wandb,
):
    table = wandb.Table(columns=['Point-Cloud', 'Segmentation-Class-Frequency', 'Model-Category', 'Split'])
    for _idx in tqdm(range(len(train_dataset))):
        _pc_viz = train_dataset[_idx].pos.numpy().tolist()
        _segmentation_label = train_dataset[_idx].y.numpy().tolist()
        _frequency_dict = {key: 0 for key in segmentation_class_frequency.keys()}
        for _label in set(_segmentation_label):
            _frequency_dict[_label] = _segmentation_label.count(_label)
        for _j in range(len(_pc_viz)):
            _pc_viz[_j] = _pc_viz[_j] + [_segmentation_label[_j] + 1 - class_offset]
        table.add_data(wandb.Object3D(np.array(_pc_viz)), _frequency_dict, config.category, 'Train-Val')
    return (table,)


@app.cell
def _(config, segmentation_class_frequency, wandb):
    _data = [[key, segmentation_class_frequency[key]] for key in segmentation_class_frequency.keys()]
    wandb.log({f'ShapeNet Class-Frequency Distribution for {config.category} Train-Val Set': wandb.plot.bar(wandb.Table(data=_data, columns=['Class', 'Frequency']), 'Class', 'Frequency', title=f'ShapeNet Class-Frequency Distribution for {config.category} Train-Val Set')})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualize Test Dataset
    """)
    return


@app.cell
def _(test_dataset, tqdm, train_dataset):
    segmentation_class_frequency_1 = {}
    for _idx in tqdm(range(len(test_dataset))):
        _pc_viz = train_dataset[_idx].pos.numpy().tolist()
        _segmentation_label = train_dataset[_idx].y.numpy().tolist()
        for _label in set(_segmentation_label):
            segmentation_class_frequency_1[_label] = _segmentation_label.count(_label)
    return (segmentation_class_frequency_1,)


@app.cell
def _(
    class_offset,
    config,
    np,
    segmentation_class_frequency_1,
    table,
    test_dataset,
    tqdm,
    train_dataset,
    wandb,
):
    for _idx in tqdm(range(len(test_dataset))):
        _pc_viz = train_dataset[_idx].pos.numpy().tolist()
        _segmentation_label = train_dataset[_idx].y.numpy().tolist()
        _frequency_dict = {key: 0 for key in segmentation_class_frequency_1.keys()}
        for _label in set(_segmentation_label):
            _frequency_dict[_label] = _segmentation_label.count(_label)
        for _j in range(len(_pc_viz)):
            _pc_viz[_j] = _pc_viz[_j] + [_segmentation_label[_j] + 1 - class_offset]
        table.add_data(wandb.Object3D(np.array(_pc_viz)), _frequency_dict, config.category, 'Test')
    wandb.log({'ShapeNet-Dataset': table})
    return


@app.cell
def _(segmentation_class_frequency_1, wandb):
    _data = [[key, segmentation_class_frequency_1[key]] for key in segmentation_class_frequency_1.keys()]
    wandb.log({f'ShapeNet Class-Frequency Distribution for Test Set': wandb.plot.bar(wandb.Table(data=_data, columns=['Class', 'Frequency']), 'Class', 'Frequency', title=f'ShapeNet Class-Frequency Distribution for Test Set')})
    return


@app.cell
def _(wandb):
    wandb.finish()
    return


if __name__ == "__main__":
    app.run()
