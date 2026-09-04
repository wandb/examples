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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pyg/point-cloud-segmentation/01_dgcnn_train.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{pyg-dgcnn-train} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🔥🔥 Train DGCNN Model using PyTorch Geometric and Weights & Biases 🪄🐝

    <!--- @wandbcode{pyg-dgcnn-train} -->

    [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wandb/examples/blob/pyg/point-cloud-segmentation/colabs/pyg/point-cloud-segmentation/01_dgcnn_train.ipynb)

    This notebook demonstrates an implementation of the [Dynamic Graph CNN](https://arxiv.org/pdf/1801.07829.pdf) for point cloud segmnetation implemented using [PyTorch Geometric](https://www.pyg.org/) and experiment tracked and visualized using [Weights & Biases](https://wandb.ai/site). The code here is inspired by [this](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/dgcnn_segmentation.py) original implementation.

    If you wish to know how to evaluate the model on the ShapeNetCore dataset using Weights & Biases, you can check out the following notebook:

    [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wandb/examples/blob/pyg/point-cloud-segmentation/colabs/pyg/point-cloud-segmentation/02_dgcnn_evaluate.ipynb)
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
    wandb_run_name = "train-dgcnn" #@param {"type": "string"}

    wandb.init(project=wandb_project, name=wandb_run_name, job_type="train")

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
    config.validation_split = 0.2
    config.batch_size = 16
    config.num_workers = 6

    config.num_nearest_neighbours = 30
    config.aggregation_operator = "max"
    config.dropout = 0.5
    config.initial_lr = 1e-3
    config.lr_scheduler_step_size = 5
    config.gamma = 0.8

    config.epochs = 1
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

    train_val_dataset = ShapeNet(
        dataset_path, config.category, split='trainval',
        transform=transform, pre_transform=pre_transform
    )
    return (train_val_dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, we need to offset the segmentation labels
    """)
    return


@app.cell
def _(tqdm, train_val_dataset):
    segmentation_class_frequency = {}
    for idx in tqdm(range(len(train_val_dataset))):
        pc_viz = train_val_dataset[idx].pos.numpy().tolist()
        segmentation_label = train_val_dataset[idx].y.numpy().tolist()
        for label in set(segmentation_label):
            segmentation_class_frequency[label] = segmentation_label.count(label)
    class_offset = min(list(segmentation_class_frequency.keys()))
    print("Class Offset:", class_offset)

    for idx in range(len(train_val_dataset)):
        train_val_dataset[idx].y -= class_offset
    return (segmentation_class_frequency,)


@app.cell
def _(config, train_val_dataset):
    num_train_examples = int((1 - config.validation_split) * len(train_val_dataset))
    train_dataset = train_val_dataset[:num_train_examples]
    val_dataset = train_val_dataset[num_train_examples:]
    return train_dataset, val_dataset


@app.cell
def _(DataLoader, config, train_dataset, val_dataset):
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers
    )
    visualization_loader = DataLoader(
        val_dataset[:10], batch_size=1,
        shuffle=False, num_workers=config.num_workers
    )
    return train_loader, val_loader, visualization_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Implementing the DGCNN Model using PyTorch Geometric
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


@app.cell
def _(DGCNN, config, device, torch, train_dataset):
    config.num_classes = train_dataset.num_classes

    model = DGCNN(
        out_channels=train_dataset.num_classes,
        k=config.num_nearest_neighbours,
        aggr=config.aggregation_operator
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.initial_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.lr_scheduler_step_size, gamma=config.gamma
    )
    return model, optimizer, scheduler


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Training DGCNN and Logging Metrics on Weights & Biases
    """)
    return


@app.cell
def _(
    F,
    ShapeNet,
    config,
    device,
    jaccard_index,
    model,
    optimizer,
    scatter,
    torch,
    tqdm,
    train_loader,
):
    def train_step(epoch):
        model.train()
    
        ious, categories = [], []
        total_loss = correct_nodes = total_nodes = 0
        y_map = torch.empty(
            train_loader.dataset.num_classes, device=device
        ).long()
        num_train_examples = len(train_loader)
    
        progress_bar = tqdm(
            train_loader, desc=f"Training Epoch {epoch}/{config.epochs}"
        )
    
        for data in progress_bar:
            data = data.to(device)
        
            optimizer.zero_grad()
            outs = model(data)
            loss = F.nll_loss(outs, data.y)
            loss.backward()
            optimizer.step()
        
            total_loss += loss.item()
        
            correct_nodes += outs.argmax(dim=1).eq(data.y).sum().item()
            total_nodes += data.num_nodes
        
            sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()
            for out, y, category in zip(outs.split(sizes), data.y.split(sizes),
                                        data.category.tolist()):
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
    
        return {
            "Train/Loss": total_loss / num_train_examples,
            "Train/Accuracy": correct_nodes / total_nodes,
            "Train/IoU": mean_iou
        }

    return (train_step,)


@app.cell
def _(
    F,
    ShapeNet,
    config,
    device,
    jaccard_index,
    model,
    scatter,
    torch,
    tqdm,
    val_loader,
):
    @torch.no_grad()
    def val_step(epoch):
        model.eval()

        ious, categories = [], []
        total_loss = correct_nodes = total_nodes = 0
        y_map = torch.empty(
            val_loader.dataset.num_classes, device=device
        ).long()
        num_val_examples = len(val_loader)
    
        progress_bar = tqdm(
            val_loader, desc=f"Validating Epoch {epoch}/{config.epochs}"
        )
    
        for data in progress_bar:
            data = data.to(device)
            outs = model(data)
        
            loss = F.nll_loss(outs, data.y)
            total_loss += loss.item()
        
            correct_nodes += outs.argmax(dim=1).eq(data.y).sum().item()
            total_nodes += data.num_nodes

            sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()
            for out, y, category in zip(outs.split(sizes), data.y.split(sizes),
                                        data.category.tolist()):
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
    
        return {
            "Validation/Loss": total_loss / num_val_examples,
            "Validation/Accuracy": correct_nodes / total_nodes,
            "Validation/IoU": mean_iou
        }

    return (val_step,)


@app.cell
def _(
    ShapeNet,
    device,
    jaccard_index,
    model,
    np,
    scatter,
    segmentation_class_frequency,
    torch,
    tqdm,
    visualization_loader,
    wandb,
):
    @torch.no_grad()
    def visualization_step(epoch, table):
        model.eval()
        for data in tqdm(visualization_loader):
            data = data.to(device)
            outs = model(data)

            predicted_labels = outs.argmax(dim=1)
            accuracy = predicted_labels.eq(data.y).sum().item() / data.num_nodes

            sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()
            ious, categories = [], []
            y_map = torch.empty(
                visualization_loader.dataset.num_classes, device=device
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
                # gt_pc_viz[j] += [segmentation_label[j] + 1 - class_offset]
                gt_pc_viz[j] += [segmentation_label[j] + 1]

            predicted_pc_viz = data.pos.cpu().numpy().tolist()
            segmentation_label = data.y.cpu().numpy().tolist()
            frequency_dict = {key: 0 for key in segmentation_class_frequency.keys()}
            for label in set(segmentation_label):
                frequency_dict[label] = segmentation_label.count(label)
            for j in range(len(predicted_pc_viz)):
                # predicted_pc_viz[j] += [segmentation_label[j] + 1 - class_offset]
                predicted_pc_viz[j] += [segmentation_label[j] + 1]

            table.add_data(
                epoch, wandb.Object3D(np.array(gt_pc_viz)),
                wandb.Object3D(np.array(predicted_pc_viz)),
                accuracy, mean_iou
            )
    
        return table

    return (visualization_step,)


@app.cell
def _(model, optimizer, torch, wandb):
    def save_checkpoint(epoch):
        """Save model checkpoints as Weights & Biases artifacts"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, "checkpoint.pt")
    
        artifact_name = wandb.util.make_artifact_name_safe(
            f"{wandb.run.name}-{wandb.run.id}-checkpoint"
        )
    
        checkpoint_artifact = wandb.Artifact(artifact_name, type="checkpoint")
        checkpoint_artifact.add_file("checkpoint.pt")
        wandb.log_artifact(
            checkpoint_artifact, aliases=["latest", f"epoch-{epoch}"]
        )

    return (save_checkpoint,)


@app.cell
def _(
    config,
    save_checkpoint,
    scheduler,
    train_step,
    val_step,
    visualization_step,
    wandb,
):
    table = wandb.Table(columns=["Epoch", "Ground-Truth", "Prediction", "Accuracy", "IoU"])

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_step(epoch)
        val_metrics = val_step(epoch)
    
        metrics = {**train_metrics, **val_metrics}
        metrics["learning_rate"] = scheduler.get_last_lr()[-1]
        wandb.log(metrics)
    
        table = visualization_step(epoch, table)
    
        scheduler.step()
        save_checkpoint(epoch)

    wandb.log({"Evaluation": table})
    return


@app.cell
def _(wandb):
    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, you can check out the following notebook to learn how to evaluate the model on the ShapeNetCore dataset using Weights & Biases, you can check out the following notebook:

    [![](https://colab.research.google.com/assets/colab-badge.svg)]()
    """)
    return


if __name__ == "__main__":
    app.run()
