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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/launch/W&B_Launch_Quickstart.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # W&B Launch Quickstart
    Use W&B Launch to quickly edit and re-run previous experiments.
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb !pip install wandb -qU
    return


@app.cell
def _():
    import wandb
    import math
    import random
    import torch, torchvision
    import torch.nn as nn
    import torchvision.transforms as T
    from tqdm.auto import tqdm

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def get_dataloader(is_train, batch_size, slice=5):
        "Get a training dataloader"
        full_dataset = torchvision.datasets.MNIST(root=".", train=is_train, transform=T.ToTensor(), download=True)
        sub_dataset = torch.utils.data.Subset(full_dataset, indices=range(0, len(full_dataset), slice))
        loader = torch.utils.data.DataLoader(dataset=sub_dataset, 
                                             batch_size=batch_size, 
                                             shuffle=True if is_train else False, 
                                             pin_memory=True, num_workers=2)
        return loader

    def get_model(dropout):
        "A simple model"
        model = nn.Sequential(nn.Flatten(),
                             nn.Linear(28*28, 256),
                             nn.BatchNorm1d(256),
                             nn.ReLU(),
                             nn.Dropout(dropout),
                             nn.Linear(256,10)).to(device)
        return model

    def validate_model(model, valid_dl, loss_func, log_images=False, batch_idx=0):
        "Compute performance of the model on the validation dataset and log a wandb.Table"
        model.eval()
        val_loss = 0.
        with torch.inference_mode():
            correct = 0
            for i, (images, labels) in tqdm(enumerate(valid_dl), leave=False):
                images, labels = images.to(device), labels.to(device)

                # Forward pass ➡
                outputs = model(images)
                val_loss += loss_func(outputs, labels)*labels.size(0)

                # Compute accuracy and accumulate
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

                # Log one batch of images to the dashboard, always same batch_idx.
                if i==batch_idx and log_images:
                    log_image_table(images, predicted, labels, outputs.softmax(dim=1))
        return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)

    def log_image_table(images, predicted, labels, probs):
        "Log a wandb.Table with (img, pred, target, scores)"
        # 🐝 Create a wandb Table to log images, labels and predictions to
        table = wandb.Table(columns=["image", "pred", "target"]+[f"score_{i}" for i in range(10)])
        for img, pred, targ, prob in zip(images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
            table.add_data(wandb.Image(img[0].numpy()*255), pred, targ, *prob.numpy())
        wandb.log({"predictions_table":table}, commit=False)

    return (
        device,
        get_dataloader,
        get_model,
        math,
        nn,
        torch,
        tqdm,
        validate_model,
        wandb,
    )


@app.cell
def _(
    device,
    get_dataloader,
    get_model,
    math,
    nn,
    torch,
    tqdm,
    validate_model,
    wandb,
):
    # 🐝 Initialise a wandb run
    wandb.init(
        project="launch-quickstart",
        config={
            "epochs": 10,
            "batch_size": 128,
            "lr": 1e-3,
            "dropout": 0.5,
            }
        )

    # Copy your config 
    config = wandb.config

    # Get the data
    train_dl = get_dataloader(is_train=True, batch_size=config.batch_size)
    valid_dl = get_dataloader(is_train=False, batch_size=2*config.batch_size)
    n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size)

    # A simple MLP model
    model = get_model(config.dropout)

    # Make the loss and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Training
    example_ct = 0
    step_ct = 0
    for epoch in tqdm(range(config.epochs)):
        model.train()
        for step, (images, labels) in enumerate(tqdm(train_dl, leave=False)):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            train_loss = loss_func(outputs, labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        
            example_ct += len(images)
            metrics = {"train/train_loss": train_loss, 
                        "train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch, 
                        "train/example_ct": example_ct}
        
            if step + 1 < n_steps_per_epoch:
                # 🐝 Log train metrics to wandb 
                wandb.log(metrics)
            
            step_ct += 1

        val_loss, accuracy = validate_model(model, valid_dl, loss_func, log_images=(epoch==(config.epochs-1)))

        # 🐝 Log train and validation metrics to wandb
        val_metrics = {"val/val_loss": val_loss, 
                        "val/val_accuracy": accuracy}
        wandb.log({**metrics, **val_metrics})
    
        print(f"Train Loss: {train_loss:.3f}, Valid Loss: {val_loss:3f}, Accuracy: {accuracy:.2f}")

        # If you had a test set, this is how you could log it as a Summary metric
        wandb.summary['test_accuracy'] = 0.8

    # 🐝 Close your wandb run 
    wandb.finish()
    return


if __name__ == "__main__":
    app.run()
