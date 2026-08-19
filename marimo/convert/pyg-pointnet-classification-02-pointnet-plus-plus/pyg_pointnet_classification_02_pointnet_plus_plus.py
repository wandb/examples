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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pyg/pointnet-classification/02_pointnet_plus_plus.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{pyg-pointnet2-train} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🔥🔥 Train PointNet++ Model using PyTorch Geometric and Weights & Biases 🪄🐝

    <!--- @wandbcode{pyg-pointnet2-train} -->

    This notebook demonstrates an implementation of the [PointeNet++](https://arxiv.org/pdf/1706.02413.pdf) architecture implemented using PyTorch Geometric and experiment tracked and visualized using [Weights & Biases](https://wandb.ai/site).

    If you wish to know how to compare and visualize the different sampling strategies used in the PointNet++ implementation, you can check out the following notebook:

    [![](https://colab.research.google.com/assets/colab-badge.svg)](http://wandb.me/pyg-pointnet2-train)
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
    # Install required packages.
    import os
    import torch
    os.environ['TORCH'] = torch.__version__
    print(torch.__version__)
    return os, torch


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
    import random
    from glob import glob
    from tqdm.auto import tqdm
    import wandb
    import torch.nn.functional as F
    import torch_geometric.transforms as T
    from torch_geometric.datasets import ModelNet
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import MLP, PointConv, fps, global_max_pool, radius

    return (
        DataLoader,
        F,
        MLP,
        ModelNet,
        PointConv,
        T,
        fps,
        glob,
        global_max_pool,
        radius,
        random,
        tqdm,
        wandb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initialize Weights & Biases

    We need to call [`wandb.init()`](https://docs.wandb.ai/ref/python/init) once at the beginning of our program to initialize a new job. This creates a new run in W&B and launches a background process to sync data.
    """)
    return


@app.cell
def _(glob, os, random, torch, wandb):
    wandb_project = "pyg-point-cloud" #@param {"type": "string"}
    wandb_run_name = "final-experiment/modelnet10/2" #@param {"type": "string"}

    wandb.init(project=wandb_project, name=wandb_run_name, job_type="baseline-train")

    # Set experiment configs to be synced with wandb
    config = wandb.config
    config.modelnet_dataset_alias = "ModelNet10" #@param ["ModelNet10", "ModelNet40"] {type:"raw"}

    config.seed = 4242 #@param {type:"number"}
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    config.sample_points = 2048 #@param {type:"slider", min:256, max:4096, step:16}

    config.categories = sorted([
        x.split(os.sep)[-2]
        for x in glob(os.path.join(
            config.modelnet_dataset_alias, "raw", '*', ''
        ))
    ])

    config.batch_size = 16 #@param {type:"slider", min:4, max:128, step:4}
    config.num_workers = 6 #@param {type:"slider", min:1, max:10, step:1}

    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(config.device)

    config.set_abstraction_ratio_1 = 0.748 #@param {type:"slider", min:0.1, max:1.0, step:0.01}
    config.set_abstraction_radius_1 = 0.4817 #@param {type:"slider", min:0.1, max:1.0, step:0.01}
    config.set_abstraction_ratio_2 = 0.3316 #@param {type:"slider", min:0.1, max:1.0, step:0.01}
    config.set_abstraction_radius_2 = 0.2447 #@param {type:"slider", min:0.1, max:1.0, step:0.01}
    config.dropout = 0.1 #@param {type:"slider", min:0.1, max:1.0, step:0.1}

    config.learning_rate = 1e-4 #@param {type:"number"}
    config.epochs = 10 #@param {type:"slider", min:1, max:100, step:1}
    config.num_visualization_samples = 20 #@param {type:"slider", min:1, max:100, step:1}
    return config, device


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load ModelNet Dataset using PyTorch Geometric

    We now load, preprocess and batch the ModelNet dataset for training, validation/testing and visualization.
    """)
    return


@app.cell
def _(DataLoader, ModelNet, T, config, random):
    pre_transform = T.NormalizeScale()
    transform = T.SamplePoints(config.sample_points)


    train_dataset = ModelNet(
        root=config.modelnet_dataset_alias,
        name=config.modelnet_dataset_alias[-2:],
        train=True,
        transform=transform,
        pre_transform=pre_transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    val_dataset = ModelNet(
        root=config.modelnet_dataset_alias,
        name=config.modelnet_dataset_alias[-2:],
        train=False,
        transform=transform,
        pre_transform=pre_transform
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    random_indices = random.sample(
        list(range(len(val_dataset))),
        config.num_visualization_samples
    )
    vizualization_loader = DataLoader(
        [val_dataset[idx] for idx in random_indices],
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers
    )
    return train_loader, val_loader, vizualization_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Implementing the PointNet++ Model using PyTorch Geometric
    """)
    return


@app.cell
def _(PointConv, fps, radius, torch):
    class SetAbstraction(torch.nn.Module):
        def __init__(self, ratio, r, nn):
            super().__init__()
            self.ratio = ratio
            self.r = r
            self.conv = PointConv(nn, add_self_loops=False)

        def forward(self, x, pos, batch):
            idx = fps(pos, batch, ratio=self.ratio)
            row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                              max_num_neighbors=64)
            edge_index = torch.stack([col, row], dim=0)
            x_dst = None if x is None else x[idx]
            x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
            pos, batch = pos[idx], batch[idx]
            return x, pos, batch

    return (SetAbstraction,)


@app.cell
def _(global_max_pool, torch):
    class GlobalSetAbstraction(torch.nn.Module):
        def __init__(self, nn):
            super().__init__()
            self.nn = nn

        def forward(self, x, pos, batch):
            x = self.nn(torch.cat([x, pos], dim=1))
            x = global_max_pool(x, batch)
            pos = pos.new_zeros((x.size(0), 3))
            batch = torch.arange(x.size(0), device=batch.device)
            return x, pos, batch

    return (GlobalSetAbstraction,)


@app.cell
def _(GlobalSetAbstraction, MLP, SetAbstraction, torch):
    class PointNet2(torch.nn.Module):
        def __init__(
            self,
            set_abstraction_ratio_1, set_abstraction_ratio_2,
            set_abstraction_radius_1, set_abstraction_radius_2, dropout
        ):
            super().__init__()

            # Input channels account for both `pos` and node features.
            self.sa1_module = SetAbstraction(
                set_abstraction_ratio_1,
                set_abstraction_radius_1,
                MLP([3, 64, 64, 128])
            )
            self.sa2_module = SetAbstraction(
                set_abstraction_ratio_2,
                set_abstraction_radius_2,
                MLP([128 + 3, 128, 128, 256])
            )
            self.sa3_module = GlobalSetAbstraction(MLP([256 + 3, 256, 512, 1024]))

            self.mlp = MLP([1024, 512, 256, 10], dropout=dropout, norm=None)

        def forward(self, data):
            sa0_out = (data.x, data.pos, data.batch)
            sa1_out = self.sa1_module(*sa0_out)
            sa2_out = self.sa2_module(*sa1_out)
            sa3_out = self.sa3_module(*sa2_out)
            x, pos, batch = sa3_out

            return self.mlp(x).log_softmax(dim=-1)

    return (PointNet2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training PointNet++ and Logging Metrics on Weights & Biases
    """)
    return


@app.cell
def _(PointNet2, config, device, torch):
    # Define PointNet++ model.
    model = PointNet2(
        config.set_abstraction_ratio_1,
        config.set_abstraction_ratio_2,
        config.set_abstraction_radius_1,
        config.set_abstraction_radius_2,
        config.dropout
    ).to(device)

    # Define Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate
    )
    return model, optimizer


@app.cell
def _(
    F,
    config,
    device,
    model,
    optimizer,
    torch,
    tqdm,
    train_loader,
    val_loader,
    vizualization_loader,
    wandb,
):
    def train_step(epoch):
        """Training Step"""
        model.train()
        epoch_loss, correct = 0, 0
        num_train_examples = len(train_loader)
    
        progress_bar = tqdm(
            range(num_train_examples),
            desc=f"Training Epoch {epoch}/{config.epochs}"
        )
        data_iter = iter(train_loader)
        for batch_idx in progress_bar:
            data = next(data_iter).to(device)
        
            optimizer.zero_grad()
            prediction = model(data)
            loss = F.nll_loss(prediction, data.y)
            loss.backward()
            optimizer.step()
        
            epoch_loss += loss.item()
            correct += prediction.max(1)[1].eq(data.y).sum().item()
    
        epoch_loss = epoch_loss / num_train_examples
        epoch_accuracy = correct / len(train_loader.dataset)
    
        wandb.log({
            "Train/Loss": epoch_loss,
            "Train/Accuracy": epoch_accuracy
        })


    def val_step(epoch):
        """Validation Step"""
        model.eval()
        epoch_loss, correct = 0, 0
        num_val_examples = len(val_loader)
    
        progress_bar = tqdm(
            range(num_val_examples),
            desc=f"Validation Epoch {epoch}/{config.epochs}"
        )
        data_iter = iter(val_loader)
        for batch_idx in progress_bar:
            data = next(data_iter).to(device)
        
            with torch.no_grad():
                prediction = model(data)
        
            loss = F.nll_loss(prediction, data.y)
            epoch_loss += loss.item()
            correct += prediction.max(1)[1].eq(data.y).sum().item()
    
        epoch_loss = epoch_loss / num_val_examples
        epoch_accuracy = correct / len(val_loader.dataset)
    
        wandb.log({
            "Validation/Loss": epoch_loss,
            "Validation/Accuracy": epoch_accuracy
        })


    def visualize_evaluation(table, epoch):
        """Visualize validation result in a Weights & Biases Table"""
        point_clouds, losses, predictions, ground_truths, is_correct = [], [], [], [], []
        progress_bar = tqdm(
            range(config.num_visualization_samples),
            desc=f"Generating Visualizations for Epoch {epoch}/{config.epochs}"
        )
    
        for idx in progress_bar:
            data = next(iter(vizualization_loader)).to(device)
        
            with torch.no_grad():
                prediction = model(data)
        
            point_clouds.append(
                wandb.Object3D(torch.squeeze(data.pos, dim=0).cpu().numpy())
            )
            losses.append(F.nll_loss(prediction, data.y).item())
            predictions.append(config.categories[int(prediction.max(1)[1].item())])
            ground_truths.append(config.categories[int(data.y.item())])
            is_correct.append(prediction.max(1)[1].eq(data.y).sum().item())
    
        table.add_data(
            epoch, point_clouds, losses, predictions, ground_truths, is_correct
        )
        return table


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

    return save_checkpoint, train_step, val_step, visualize_evaluation


@app.cell
def _(
    config,
    save_checkpoint,
    train_step,
    val_step,
    visualize_evaluation,
    wandb,
):
    table = wandb.Table(
        columns=[
            "Epoch",
            "Point-Clouds",
            "Losses",
            "Predicted-Classes",
            "Ground-Truth",
            "Is-Correct"
        ]
    )
    for epoch in range(1, config.epochs + 1):
        train_step(epoch)
        val_step(epoch)
        visualize_evaluation(table, epoch)
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
    Next, you can check out the following notebook to learn how to run a hyperparameter sweep on our PointNet++ trainig loop using Weights & Biases:

    |Tune Hyperparameters using Weights & Biases Sweep|[![](https://colab.research.google.com/assets/colab-badge.svg)](http://wandb.me/pyg-pointnet2-sweep)|
    """)
    return


if __name__ == "__main__":
    app.run()
