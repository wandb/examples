# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.9",
#     "torch>=2.1",
#     "torchvision>=0.16",
#     "wandb>=0.18",
#     "tqdm",
# ]
# ///
"""Train an MNIST CNN with PyTorch, track the run with Weights & Biases,
and link the resulting model artifact to a W&B Registry collection.

Run:

    uvx marimo edit mnist_registry.py --sandbox

This notebook is interactive: hyperparameters live in a form, and training is
gated by a button so slider changes do not trigger runs.
"""

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium", app_title="MNIST -> W&B Registry")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        """
        # MNIST -> W&B Run -> Registry

        ## What you will build

        - A **W&B run** with training and test metrics, gradient histograms,
          and example test-set predictions logged as images.
        - A **model Artifact** named `mnist-cnn-<run-id>` of type `model`,
          carrying metadata (test accuracy, parameter count, hyperparameters).
        - A version of that Artifact **linked into a W&B Registry collection**
          so it appears under registered models org-wide.

        ## Prerequisites

        - **`wandb login`** completed in your shell before starting marimo.
          This notebook will not prompt for an API key interactively.
        - A W&B entity (your user or a team) the run will be written to.
        - A **W&B Registry** must exist in your org. The built-in Model
          registry is provisioned automatically in newer orgs. If linking
          fails, the Registry step surfaces remediation guidance inline
          instead of crashing.
        - A GPU is optional. The defaults are tuned to finish in ~2 minutes
          on CPU.
        """
    )
    return (mo,)


@app.cell
def _():
    import os

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    import wandb
    from tqdm.auto import tqdm

    return (
        DataLoader,
        F,
        datasets,
        nn,
        optim,
        os,
        torch,
        tqdm,
        transforms,
        wandb,
    )


@app.cell
def _(mo, torch):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_note = "CUDA GPU detected. Training will be fast."
        callout_kind = "success"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_note = "Apple MPS detected. Training will run on the GPU."
        callout_kind = "success"
    else:
        device = torch.device("cpu")
        device_note = (
            "No GPU detected. Training will run on CPU. With the default "
            "hyperparameters this takes ~2 minutes."
        )
        callout_kind = "warn"

    mo.callout(mo.md(f"**Device:** `{device}` &mdash; {device_note}"), kind=callout_kind)
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Hyperparameters

        Configure the training run and the Registry target. The defaults reach
        roughly 98% test accuracy in about two minutes on CPU. The **Registry**
        section controls where the trained model is linked after training
        finishes.
        """
    )
    return


@app.cell
def _(mo):
    epochs = mo.ui.slider(start=1, stop=10, step=1, value=3, label="Epochs")
    batch_size = mo.ui.dropdown(
        options=["32", "64", "128", "256"], value="64", label="Batch size"
    )
    lr = mo.ui.slider(
        start=0.001, stop=0.1, step=0.001, value=0.01, label="Learning rate", show_value=True
    )
    momentum = mo.ui.slider(
        start=0.0, stop=0.99, step=0.01, value=0.5, label="SGD momentum", show_value=True
    )
    seed = mo.ui.number(start=0, stop=99999, value=42, label="Random seed")

    project = mo.ui.text(value="marimo-mnist-registry", label="W&B project")
    entity = mo.ui.text(value="", label="W&B entity (blank uses your default)")
    run_name = mo.ui.text(value="", label="Run name (blank auto-generates)")

    registry_name = mo.ui.text(value="model", label="W&B Registry name")
    collection_name = mo.ui.text(value="MNIST Classifiers", label="Registry collection")
    link_to_registry = mo.ui.checkbox(value=True, label="Link artifact to Registry")

    form = mo.vstack(
        [
            mo.md("### Training"),
            mo.hstack([epochs, batch_size]),
            mo.hstack([lr, momentum]),
            seed,
            mo.md("### W&B run"),
            mo.hstack([project, entity, run_name]),
            mo.md("### Registry"),
            mo.hstack([registry_name, collection_name, link_to_registry]),
        ]
    )
    form
    return (
        batch_size,
        collection_name,
        entity,
        epochs,
        link_to_registry,
        lr,
        momentum,
        project,
        registry_name,
        run_name,
        seed,
    )


@app.cell
def _(
    batch_size,
    collection_name,
    entity,
    epochs,
    link_to_registry,
    lr,
    momentum,
    project,
    registry_name,
    run_name,
    seed,
):
    config = {
        "epochs": epochs.value,
        "batch_size": int(batch_size.value),
        "lr": lr.value,
        "momentum": momentum.value,
        "seed": seed.value,
        "architecture": "CNN",
        "dataset": "MNIST",
    }
    wandb_project = project.value or None
    wandb_entity = entity.value or None
    wandb_run_name = run_name.value or None
    registry_name_v = registry_name.value.strip()
    collection_name_v = collection_name.value.strip()
    link_to_registry_v = link_to_registry.value
    return (
        collection_name_v,
        config,
        link_to_registry_v,
        registry_name_v,
        wandb_entity,
        wandb_project,
        wandb_run_name,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Train

        Training is gated by an explicit button click so changing a
        hyperparameter does not by itself start a run. Click **Train model**
        to begin. Once the first run has completed, clicking the button again
        starts a new run with whatever the form values are at that moment;
        the previous run is finished cleanly first.
        """
    )
    return


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Train model", kind="success")
    train_button
    return (train_button,)


@app.cell
def _(F, nn):
    class Net(nn.Module):
        """The same small CNN used in examples/pytorch/pytorch-cnn-mnist.

        Two convolutional layers (10 and 20 filters, 5x5 kernels) feed into two
        fully connected layers (50 hidden units, 10 outputs). Roughly 21k
        parameters.
        """

        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    return (Net,)


@app.cell
def _(DataLoader, batch_size, datasets, device, mo, transforms):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)

    # Only batch_size and device affect the loaders, so we depend on them
    # directly rather than the full config dict; this avoids re-creating the
    # loaders whenever an unrelated hyperparameter changes.
    bs = int(batch_size.value)
    loader_kwargs = (
        {"num_workers": 2, "pin_memory": True} if device.type == "cuda" else {}
    )
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, **loader_kwargs
    )
    test_loader = DataLoader(
        test_ds, batch_size=1000, shuffle=False, **loader_kwargs
    )

    mo.md(
        f"**Train:** {len(train_ds):,} examples &middot; "
        f"**Test:** {len(test_ds):,} examples &middot; "
        f"**Batch size:** {bs}"
    )
    return test_ds, test_loader, train_ds, train_loader


@app.cell
def _(
    Net,
    config,
    device,
    mo,
    optim,
    torch,
    train_button,
    wandb,
    wandb_entity,
    wandb_project,
    wandb_run_name,
):
    mo.stop(not train_button.value, mo.md("Click **Train model** to begin."))

    # Defensive: finish any prior run still attached to this Python process.
    # marimo keeps the kernel alive across re-clicks, so a second click — or
    # a slider change after the first click — re-executes this cell. Without
    # this guard `wandb.init` would warn about a run already being active.
    # `wandb.finish` blocks until the prior run's tail logs are uploaded.
    if wandb.run is not None:
        wandb.finish()

    torch.manual_seed(config["seed"])

    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=wandb_run_name,
        config=config,
        job_type="train",
    )

    # Use `epoch` as the x-axis for train and test metrics in the W&B UI.
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("test/*", step_metric="epoch")

    model = Net().to(device)
    # `log="gradients"` is the conventional choice for didactic examples;
    # `log="all"` would additionally log parameter histograms at extra cost.
    wandb.watch(model, log="gradients", log_freq=100)
    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"]
    )

    mo.md(f"Run started: [`{run.name}`]({run.url})")
    return model, optimizer, run


@app.cell
def _(
    F,
    config,
    device,
    mo,
    model,
    optimizer,
    test_loader,
    torch,
    tqdm,
    train_button,
    train_loader,
    wandb,
):
    mo.stop(not train_button.value, mo.md(""))

    history = []
    best_acc = 0.0
    final_acc = 0.0
    final_loss = 0.0

    for epoch in range(1, config["epochs"] + 1):
        # ---- train ----
        model.train()
        for batch_idx, (data, target) in enumerate(
            tqdm(train_loader, desc=f"epoch {epoch}/{config['epochs']}")
        ):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 50 == 0:
                wandb.log({"train/loss": loss.item(), "epoch": epoch})

        # ---- test ----
        model.eval()
        test_loss = 0.0
        correct = 0
        example_images = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                # Pull up to 16 example predictions from the first batch we see.
                while len(example_images) < 16 and len(example_images) < data.size(0):
                    j = len(example_images)
                    example_images.append(
                        wandb.Image(
                            data[j],
                            caption=(
                                f"pred={pred[j].item()} "
                                f"true={target[j].item()}"
                            ),
                        )
                    )

        test_loss /= len(test_loader.dataset)
        test_acc = correct / len(test_loader.dataset)
        best_acc = max(best_acc, test_acc)
        final_acc = test_acc
        final_loss = test_loss
        wandb.log(
            {
                "test/loss": test_loss,
                "test/accuracy": test_acc,
                "epoch": epoch,
                "examples": example_images,
            }
        )
        history.append(
            {"epoch": epoch, "test_loss": test_loss, "test_acc": test_acc}
        )

    return best_acc, final_acc, final_loss, history


@app.cell
def _(final_acc, final_loss, history, mo, train_button):
    mo.stop(not train_button.value, mo.md(""))
    mo.vstack(
        [
            mo.md("### Training summary"),
            mo.ui.table(history, selection=None),
            mo.md(
                f"**Final test accuracy:** {final_acc:.2%} &middot; "
                f"**Final test loss:** {final_loss:.4f}"
            ),
        ]
    )
    return


@app.cell
def _(mo, model, os, torch, train_button):
    mo.stop(not train_button.value, mo.md(""))

    model_path = "mnist_cnn.pt"
    torch.save(model.state_dict(), model_path)

    mo.md(
        f"Saved `{model_path}` ({os.path.getsize(model_path) / 1024:.1f} KB)"
    )
    return (model_path,)


@app.cell
def _(
    best_acc,
    config,
    final_acc,
    mo,
    model,
    model_path,
    run,
    test_ds,
    train_button,
    train_ds,
    wandb,
):
    mo.stop(not train_button.value, mo.md(""))

    num_params = sum(p.numel() for p in model.parameters())

    artifact = wandb.Artifact(
        name=f"mnist-cnn-{run.id}",
        type="model",
        description=(
            "Small CNN trained on MNIST. Architecture: 2 conv layers "
            "(10 and 20 filters, 5x5 kernels) + 2 FC layers (50, 10)."
        ),
        metadata={
            "framework": "pytorch",
            "architecture": "CNN",
            "num_parameters": num_params,
            "dataset": "MNIST",
            "train_size": len(train_ds),
            "test_size": len(test_ds),
            "test_accuracy": final_acc,
            "best_test_accuracy": best_acc,
            "hyperparameters": dict(config),
        },
    )
    artifact.add_file(model_path)

    # We only log a single artifact per run (the final-epoch weights), so we
    # tag it `latest` unconditionally. Use the Registry UI or the API to
    # promote a specific version with aliases like `best` or `production`
    # after comparing across runs.
    aliases = ["latest"]

    logged = run.log_artifact(artifact, aliases=aliases)
    # Block until the artifact has fully committed server-side. Without this,
    # link_artifact below may race on the version reference.
    logged.wait()

    mo.md(
        f"Artifact logged: `{artifact.name}` with aliases `{aliases}`"
    )
    return (logged,)


@app.cell
def _(
    collection_name_v,
    link_to_registry_v,
    logged,
    mo,
    registry_name_v,
    run,
    train_button,
):
    mo.stop(not train_button.value, mo.md(""))
    mo.stop(
        not link_to_registry_v,
        mo.md(
            "_Registry linking is disabled (checkbox unchecked). "
            "The artifact is logged to the run but not linked to a Registry "
            "collection._"
        ),
    )

    target_path = f"wandb-registry-{registry_name_v}/{collection_name_v}"

    try:
        run.link_artifact(artifact=logged, target_path=target_path)
        link_result = mo.callout(
            mo.md(
                f"**Linked to Registry:** `{target_path}`\n\n"
                f"Open the Registry at "
                f"[https://wandb.ai/registry](https://wandb.ai/registry) to "
                f"see the version."
            ),
            kind="success",
        )
    except Exception as exc:  # noqa: BLE001 - we want to surface any failure to the reader
        link_result = mo.callout(
            mo.md(
                f"**Registry link failed.**\n\n"
                f"Target path: `{target_path}`\n\n"
                f"Error: `{exc}`\n\n"
                f"Common causes:\n\n"
                f"- The Registry `{registry_name_v}` does not exist in your "
                f"org. An org admin can create the Model registry from the "
                f"W&B Registry UI.\n"
                f"- Your account lacks Registry write permission.\n"
                f"- Your org is on the legacy Model Registry. In that case "
                f"use the legacy pattern:\n\n"
                f"  ```python\n"
                f"  run.link_artifact(\n"
                f"      logged,\n"
                f"      target_path='model-registry/{collection_name_v}',\n"
                f"  )\n"
                f"  ```"
            ),
            kind="danger",
        )
    link_result
    return


@app.cell(hide_code=True)
def _(collection_name_v, mo, registry_name_v, run, train_button):
    mo.stop(not train_button.value, mo.md(""))
    mo.md(
        f"""
        ## Verify

        1. Open the run page: [{run.name}]({run.url}). Confirm the
           **Charts**, **System**, and **Examples** panels populated.
        2. Click **Artifacts** in the run's left nav. Confirm the
           `mnist-cnn-{run.id}` model artifact is listed with metadata
           (test accuracy, hyperparameters, number of parameters).
        3. Go to [wandb.ai/registry](https://wandb.ai/registry), open the
           **{registry_name_v.title()}** registry, then the
           **{collection_name_v}** collection. Confirm the linked version
           is present.

        ## Consume the registered model

        From any other script or notebook, fetch the latest registered version:

        ```python
        import wandb
        api = wandb.Api()
        art = api.artifact(
            "wandb-registry-{registry_name_v}/{collection_name_v}:latest"
        )
        art.download()  # writes mnist_cnn.pt under ./artifacts/.../
        ```

        ## Next steps

        - **Promote a version.** From the Registry UI, add the `production`
          alias to the version you want consumers to pick up. The same
          collection path with `:production` will then resolve to it.
        - **Compare runs.** Re-run with a deeper architecture or a different
          learning rate. Group runs in the W&B UI to compare test accuracy
          across configurations.
        - **Automate on promotion.** Configure a W&B Automation on the
          collection to trigger evaluation jobs or webhooks when a new
          version is linked.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Finish

        Closes the W&B run so the run summary and the Registry version
        finalize on the server.
        """
    )
    return


@app.cell
def _(mo, train_button, wandb):
    mo.stop(not train_button.value, mo.md(""))
    # Mirror of the defensive `wandb.finish` at the top of the training cell:
    # leaves the kernel in a clean state for the next Train click.
    if wandb.run is not None:
        wandb.finish()
    mo.md("Run finished. Click **Train model** again to start a new run.")
    return


if __name__ == "__main__":
    app.run()
