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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pyg/pointnet-classification/03_sweep.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{pyg-pointnet2-sweep} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🔥🔥 Run a Hyperparamter Sweep on PointNet++ 🪄🐝

    <!--- @wandbcode{pyg-pointnet2-sweep} -->

    This notebook demonstrates the process of running a [Hyperparameter Sweep using Weights & Biases](https://docs.wandb.ai/guides/sweeps) on our point cloud classification training workflow in order to maximize the performance of our model.

    If you wish to know how to implement the PointNet++ architecture and train it you can check out the following [notebook](http://wandb.me/pyg-sampling).
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
    import gc
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
        gc,
        glob,
        global_max_pool,
        radius,
        tqdm,
        wandb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Function to Build Data Loaders
    """)
    return


@app.cell
def _(DataLoader, ModelNet, T):
    def get_dataset_and_loaders(sample_points, batch_size, num_workers):
        pre_transform = T.NormalizeScale()
        transform = T.SamplePoints(sample_points)

        train_dataset = ModelNet(
            root="ModelNet10",
            name='10',
            train=True,
            transform=transform,
            pre_transform=pre_transform
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        val_dataset = ModelNet(
            root="ModelNet10",
            name='10',
            train=False,
            transform=transform,
            pre_transform=pre_transform
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    
        return train_dataset, train_loader, val_dataset, val_loader

    return (get_dataset_and_loaders,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Implementing the PointNet++ Model using PyTorch Geometric
    """)
    return


@app.cell
def _(MLP, PointConv, fps, global_max_pool, radius, torch):
    class SetAbstraction(torch.nn.Module):
        def __init__(self, ratio, r, nn):
            super().__init__()
            self.ratio = ratio
            self.r = r
            self.conv = PointConv(nn, add_self_loops=False)

        def forward(self, x, pos, batch):
            idx = fps(pos, batch, ratio=self.ratio)
            row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
            edge_index = torch.stack([col, row], dim=0)
            x_dst = None if x is None else x[idx]
            x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
            pos, batch = pos[idx], batch[idx]
            return x, pos, batch


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


    class PointNet2(torch.nn.Module):
        def __init__(self, set_abstraction_ratio_1, set_abstraction_ratio_2, dropout):
            super().__init__()

            # Input channels account for both `pos` and node features.
            self.sa1_module = SetAbstraction(
                set_abstraction_ratio_1, 0.2, MLP([3, 64, 64, 128])
            )
            self.sa2_module = SetAbstraction(
                set_abstraction_ratio_2, 0.4, MLP([128 + 3, 128, 128, 256])
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
    ## Define a Training Function Instrumented with WandB
    """)
    return


@app.cell
def _(F, PointNet2, gc, get_dataset_and_loaders, glob, os, torch, tqdm, wandb):
    def train():
        wandb.init(project="pyg-point-cloud", entity="geekyrakshit")
    
        # Set Default Configs
        config = wandb.config
        config.categories = sorted([
            x.split(os.sep)[-2]
            for x in glob(os.path.join("ModelNet10", "raw", '*', ''))
        ])
        config.num_workers = 6
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(config.device)
        config.learning_rate = 1e-4
        config.epochs = 5
    
        # Get tuned configs from sweep
        batch_size = config.batch_size
        sample_points = config.sample_points
        set_abstraction_ratio_1 = config.set_abstraction_ratio_1
        set_abstraction_ratio_2 = config.set_abstraction_ratio_2
        dropout = config.dropout
    
        # Create datasets and dataloaders
        (
            train_dataset, train_loader, val_dataset, val_loader
        ) = get_dataset_and_loaders(
            sample_points, batch_size, config.num_workers
        )
    
        model = PointNet2(
            set_abstraction_ratio_1, set_abstraction_ratio_2, dropout
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate
        )
    
        for epoch in range(1, config.epochs + 1):
        
            # Training Step
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
        
            # Validation Step
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
        
            # Save Checkpoint
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
    
        model = model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    return (train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Start the Hyperparameter Sweep
    """)
    return


@app.cell
def _(train, wandb):
    # Define sweep configuration
    sweep_configuration = {
        'method': 'bayes',
        'metric': {'goal': 'maximize', 'name': 'Validation/Accuracy'},
        'parameters': 
        {
            'batch_size': {'values': [8, 16, 32, 64]},
            'sample_points': {'values': [512, 1024, 2048]},
            'set_abstraction_ratio_1': {'min': 0.1, 'max': 0.9},
            'set_abstraction_ratio_2': {'min': 0.1, 'max': 0.9},
            'dropout': {'min': 0.1, 'max': 0.7},
         }
    }

    # Get Sweep ID
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, project='pyg-point-cloud', entity="geekyrakshit"
    )

    # Run Sweep
    wandb.agent(sweep_id, function=train, count=30)
    return


if __name__ == "__main__":
    app.run()
