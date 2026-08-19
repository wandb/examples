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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pyg/point-cloud-segmentation/02_dgcnn_evaluate.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{pyg-dgcnn-eval} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🔥🔥 Evaluate DGCNN Model Weights & Biases 🪄🐝

    <!--- @wandbcode{pyg-dgcnn-eval} -->

    [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wandb/examples/blob/pyg/point-cloud-segmentation/colabs/pyg/point-cloud-segmentation/02_dgcnn_evaluate.ipynb)

    This notebook demonstrates the evaluation of [Dynamic Graph CNN](https://arxiv.org/pdf/1801.07829.pdf) for point cloud segmnetation. You can check the following notebook for referring to the training code:

    [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wandb/examples/blob/pyg/point-cloud-segmentation/colabs/pyg/point-cloud-segmentation/01_dgcnn_train.ipynb)
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
    return os, torch


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
    import random
    import numpy as np
    from tqdm.auto import tqdm
    import torch.nn.functional as F
    from torch_scatter import scatter
    from torchmetrics.functional import jaccard_index
    import torch_geometric.transforms as T
    from torch_geometric.datasets import ShapeNet
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import MLP, DynamicEdgeConv

    return (
        DataLoader,
        DynamicEdgeConv,
        F,
        MLP,
        ShapeNet,
        T,
        jaccard_index,
        np,
        random,
        scatter,
        tqdm,
        wandb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Initialize Weights & Biases

    We need to call [`wandb.init()`](https://docs.wandb.ai/ref/python/init) once at the beginning of our program to initialize a new job. This creates a new run in W&B and launches a background process to sync data.
    """)
    return


@app.cell
def _(random, torch, wandb):
    wandb_project = "pyg-point-cloud" #@param {"type": "string"}
    wandb_run_name = "evaluate-dgcnn" #@param {"type": "string"}

    wandb.init(project=wandb_project, name=wandb_run_name, job_type="evaluate")

    config = wandb.config

    config.seed = 42
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    config.category = 'Airplane' #@param ["Bag", "Cap", "Car", "Chair", "Earphone", "Guitar", "Knife", "Lamp", "Laptop", "Motorbike", "Mug", "Pistol", "Rocket", "Skateboard", "Table"] {type:"raw"}
    config.random_jitter_translation = 1e-2
    config.random_rotation_interval_x = 15
    config.random_rotation_interval_y = 15
    config.random_rotation_interval_z = 15
    config.batch_size = 1
    config.num_workers = 6

    config.num_nearest_neighbours = 30
    config.aggregation_operator = "max"
    config.dropout = 0.5
    config.initial_lr = 1e-3
    config.lr_scheduler_step_size = 20
    config.gamma = 0.8

    config.artifact_address = 'wandb/point-cloud-segmentation/dgcnn-3n97rfrv-checkpoint:v29'
    config.epochs = 30
    return config, device


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Load ShapeNet Dataset using PyTorch Geometric

    We now load, preprocess and batch the ModelNet dataset for training, validation/testing and visualization.
    """)
    return


@app.cell
def _(T, config):
    transform = T.Compose([
        T.RandomJitter(config.random_jitter_translation),
        T.RandomRotate(config.random_rotation_interval_x, axis=0),
        T.RandomRotate(config.random_rotation_interval_y, axis=1),
        T.RandomRotate(config.random_rotation_interval_z, axis=2)
    ])
    pre_transform = T.NormalizeScale()
    return pre_transform, transform


@app.cell
def _(ShapeNet, config, os, pre_transform, transform):
    dataset_path = os.path.join('ShapeNet', config.category)

    train_dataset = ShapeNet(
        dataset_path, config.category, split='trainval',
        transform=transform, pre_transform=pre_transform
    )
    test_dataset = ShapeNet(
        dataset_path, config.category, split='test',
        pre_transform=pre_transform
    )
    return test_dataset, train_dataset


@app.cell
def _(test_dataset, tqdm, train_dataset):
    segmentation_class_frequency = {}
    for _idx in tqdm(range(len(train_dataset))):
        pc_viz = train_dataset[_idx].pos.numpy().tolist()
        segmentation_label = train_dataset[_idx].y.numpy().tolist()
        for label in set(segmentation_label):
            segmentation_class_frequency[label] = segmentation_label.count(label)
    for _idx in tqdm(range(len(test_dataset))):
        pc_viz = train_dataset[_idx].pos.numpy().tolist()
        segmentation_label = train_dataset[_idx].y.numpy().tolist()
        for label in set(segmentation_label):
            segmentation_class_frequency[label] = segmentation_label.count(label)
    class_offset = min(list(segmentation_class_frequency.keys()))
    class_offset
    return class_offset, segmentation_class_frequency


@app.cell
def _(class_offset, test_dataset, tqdm, train_dataset):
    for _idx in tqdm(range(len(train_dataset))):
        train_dataset[_idx].y -= class_offset
    for _idx in tqdm(range(len(test_dataset))):
        test_dataset[_idx].y -= class_offset
    return


@app.cell
def _(DataLoader, config, test_dataset, train_dataset):
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers
    )
    return test_loader, train_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Load Checkpoint
    """)
    return


@app.cell
def _(DynamicEdgeConv, F, MLP, torch):
    class DGCNN(torch.nn.Module):
        def __init__(self, out_channels, k=30, aggr='max'):
            super().__init__()

            self.conv1 = DynamicEdgeConv(MLP([2 * 6, 64, 64]), k, aggr)
            self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)
            self.conv3 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)

            self.mlp = MLP(
                [3 * 64, 1024, 256, 128, out_channels],
                dropout=0.5, norm=None
            )

        def forward(self, data):
            x, pos, batch = data.x, data.pos, data.batch
            x0 = torch.cat([x, pos], dim=-1)
        
            x1 = self.conv1(x0, batch)
            x2 = self.conv2(x1, batch)
            x3 = self.conv3(x2, batch)
        
            out = self.mlp(torch.cat([x1, x2, x3], dim=1))
            return F.log_softmax(out, dim=1)

    return (DGCNN,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since we saved the checkpoints as artifacts on our Weights & Biases workspace, we can now fetch and load them.
    """)
    return


@app.cell
def _(DGCNN, config, device, os, torch, train_dataset, wandb):
    config.num_classes = train_dataset.num_classes

    model = DGCNN(
        out_channels=train_dataset.num_classes,
        k=config.num_nearest_neighbours,
        aggr=config.aggregation_operator
    ).to(device)

    model_artifact = wandb.use_artifact(config.artifact_address, type='checkpoint')
    artifact_dir = model_artifact.download()
    model_checkpoint_path = os.path.join(artifact_dir, "checkpoint.pt")

    model.load_state_dict(torch.load(model_checkpoint_path)["model_state_dict"])
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Evaluation
    """)
    return


@app.cell
def _(
    ShapeNet,
    class_offset,
    device,
    jaccard_index,
    model,
    np,
    scatter,
    segmentation_class_frequency,
    torch,
    tqdm,
    wandb,
):
    def evaluate(loader, split, table):
        total_accuracy, total_iou = 0, 0
        for data in tqdm(loader):
            data = data.to(device)
            with torch.no_grad():
                model.eval()
                outs = model(data)

                predicted_labels = outs.argmax(dim=1)
                accuracy = predicted_labels.eq(data.y).sum().item() / data.num_nodes

                sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()
                ious, categories = [], []
                y_map = torch.empty(
                    loader.dataset.num_classes, device=device
                ).long()
                for out, y, category in zip(
                    outs.split(sizes), data.y.split(sizes), data.category.tolist()
                ):
                    category = list(ShapeNet.seg_classes.keys())[category]
                    part = ShapeNet.seg_classes[category]
                    part = torch.tensor(part, device=device)
                    y_map[part] = torch.arange(part.size(0), device=device)
                    iou = jaccard_index(
                        out[:, part].argmax(dim=-1), y_map[y],
                        task="multiclass", num_classes=part.size(0)
                    )
                    ious.append(iou)
                categories.append(data.category)
                iou = torch.tensor(ious, device=device)
                category = torch.cat(categories, dim=0)
                mean_iou = float(scatter(iou, category, reduce='mean').mean())

                gt_pc_viz = data.pos.cpu().numpy().tolist()
                segmentation_label = data.y.cpu().numpy().tolist()
                frequency_dict = {key: 0 for key in segmentation_class_frequency.keys()}
                for label in set(segmentation_label):
                    frequency_dict[label] = segmentation_label.count(label)
                for j in range(len(gt_pc_viz)):
                    gt_pc_viz[j] += [segmentation_label[j] + 1 - class_offset]

                predicted_pc_viz = data.pos.cpu().numpy().tolist()
                segmentation_label = data.y.cpu().numpy().tolist()
                frequency_dict = {key: 0 for key in segmentation_class_frequency.keys()}
                for label in set(segmentation_label):
                    frequency_dict[label] = segmentation_label.count(label)
                for j in range(len(predicted_pc_viz)):
                    predicted_pc_viz[j] += [segmentation_label[j] + 1 - class_offset]

                table.add_data(
                    wandb.Object3D(np.array(gt_pc_viz)),
                    wandb.Object3D(np.array(predicted_pc_viz)),
                    accuracy, mean_iou, split, "DGCNN"
                )
                total_accuracy += accuracy
                total_iou += mean_iou
    
        wandb.log({
            f"{split}/Accuracy": total_accuracy / len(loader),
            f"{split}/IoU": total_iou / len(loader),
        })
    
        return table

    return (evaluate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We evaluate the results and store them in a Weights & Biases Table.
    """)
    return


@app.cell
def _(evaluate, test_loader, train_loader, wandb):
    table = wandb.Table(columns=["Ground-Truth", "Prediction", "Accuracy", "IoU", "Split", "Model-Name"])
    evaluate(train_loader, "Train-Val", table)
    evaluate(test_loader, "Test", table)
    return (table,)


@app.cell
def _(table, wandb):
    wandb.log({"Evaluation-Results": table})
    return


@app.cell
def _(wandb):
    wandb.finish()
    return


if __name__ == "__main__":
    app.run()
