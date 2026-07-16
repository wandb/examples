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

The notebook has three interactive cells: fill in the form, click **Train
model**, then read the results. Everything between the inputs and the button
runs as a single step, so one click trains, logs, saves, and registers.
"""

import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium", app_title="MNIST -> W&B Registry")

with app.setup(hide_code=True):
    import marimo as mo
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    import wandb
    from tqdm.auto import tqdm

    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_note = "CUDA GPU detected. Training will be fast."
        device_kind = "success"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_note = "Apple MPS detected. Training will run on the GPU."
        device_kind = "success"
    else:
        device = torch.device("cpu")
        device_note = (
            "No GPU detected. Training will run on CPU. With the default "
            "hyperparameters this takes about 2 minutes."
        )
        device_kind = "warn"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # MNIST -> W&B Run -> Registry

    ## What you will build

    - A **W&B run** with a single **Training** section charting loss and
      accuracy.
    - A **model Artifact** named `mnist-cnn-<run-id>` of type `model`,
      carrying metadata (test accuracy, parameter count, hyperparameters).
    - A version of that Artifact **linked into a W&B Registry collection**
      so it appears under registered models org-wide.

    ## Prerequisites

    - Authenticate with W&B one of two ways: run **`wandb login`** in
      your shell before starting marimo, or paste your key into the
      **W&B API key** field in the form below. Get your key from
      [wandb.ai/authorize](https://wandb.ai/authorize).
    - A W&B **team** to write the run to, set in the **W&B entity** field.
      Accounts created after May 2024 have no personal entity, so the run
      must go to a team — your username will not work as an entity.
    - A **W&B Registry** must exist in your org, and your account needs at
      least the **Member** role on it (linking an artifact is a write
      action). The built-in Model registry is provisioned automatically in
      newer orgs. If linking fails (for example, from a view-only seat),
      the run still completes and the Registry step explains how to fix it.
    - A GPU is optional. The defaults finish in about 2 minutes on CPU.
    """)
    return


@app.cell
def _():
    mo.outline()
    return


@app.cell(hide_code=True)
def _():
    mo.callout(
        mo.md(f"**Device:** `{device}`. {device_note}"),
        kind=device_kind,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Configuration

    Set the hyperparameters and W&B targets, then click **Train model** below.
    """)
    return


@app.cell(hide_code=True)
def _():
    epochs = mo.ui.slider(start=1, stop=10, step=1, value=3, label="Epochs")
    batch_size = mo.ui.dropdown(
        options=["32", "64", "128", "256"], value="64", label="Batch size"
    )
    lr = mo.ui.slider(
        start=0.001,
        stop=0.1,
        step=0.001,
        value=0.01,
        label="Learning rate",
        show_value=True,
    )
    momentum = mo.ui.slider(
        start=0.0,
        stop=0.99,
        step=0.01,
        value=0.5,
        label="SGD momentum",
        show_value=True,
    )
    seed = mo.ui.number(start=0, stop=99999, value=42, label="Random seed")

    project = mo.ui.text(value="marimo-mnist-registry", label="W&B project")
    entity = mo.ui.text(
        value="",
        label="W&B entity \u2014 a team you belong to (blank uses your default)",
    )
    run_name = mo.ui.text(value="", label="Run name (blank auto-generates)")
    api_key = mo.ui.text(
        value="",
        kind="password",
        label="W&B API key (blank uses your shell login)",
    )

    registry_name = mo.ui.text(value="model", label="W&B Registry name")
    collection_name = mo.ui.text(
        value="MNIST Classifiers", label="Registry collection"
    )
    link_to_registry = mo.ui.checkbox(
        value=True, label="Link artifact to Registry"
    )

    # Batch every control into one form so training only kicks off on submit.
    # `form.value` is None until the user clicks Train model, then becomes a dict
    # keyed by the names below \u2014 the training cell gates on that.
    form = (
        mo.md(
            """
            **Training.**

            {epochs}  {batch_size}

            {lr}  {momentum}

            {seed}

            **W&B run.**

            {api_key}

            {project}

            {entity}

            {run_name}

            **Registry.**

            {registry_name}  {collection_name}  {link_to_registry}
            """
        )
        .batch(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            momentum=momentum,
            seed=seed,
            api_key=api_key,
            project=project,
            entity=entity,
            run_name=run_name,
            registry_name=registry_name,
            collection_name=collection_name,
            link_to_registry=link_to_registry,
        )
        .form(submit_button_label="Train model", bordered=False)
    )

    form
    return (form,)


@app.class_definition
class Net(nn.Module):
    """Small CNN: 2 conv layers (10, 20 filters, 5x5) + 2 FC (50, 10).

    Defined in its own cell so the training cell and the consume cell can
    share it (marimo forbids defining the same name in two cells).
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


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Training
    """)
    return


@app.cell
def _(form):
    mo.stop(
        form.value is None,
        mo.md(
            "Training hasn't started yet. Fill in the form above and click "
            "**Train model** to start the run — it trains the model, logs loss "
            "and accuracy, saves the weights as an Artifact, links them to the "
            "Registry, and classifies a few test digits."
        ),
    )

    cfg = form.value
    config = {
        "epochs": cfg["epochs"],
        "batch_size": int(cfg["batch_size"]),
        "lr": cfg["lr"],
        "momentum": cfg["momentum"],
        "seed": cfg["seed"],
        "architecture": "CNN",
        "dataset": "MNIST",
    }
    registry_name_v = cfg["registry_name"].strip()
    collection_name_v = cfg["collection_name"].strip()

    # Authenticate and start the run. Finish any prior run first (marimo keeps the
    # kernel alive across re-submits). A key pasted into the form wins; otherwise
    # fall back to ambient login (shell `wandb login`, WANDB_API_KEY, or netrc).
    # The key is never written to the run config.
    if wandb.run is not None:
        wandb.finish()
    if cfg["api_key"]:
        wandb.login(key=cfg["api_key"])
    torch.manual_seed(config["seed"])

    try:
        run = wandb.init(
            project=cfg["project"] or None,
            entity=cfg["entity"] or None,
            name=cfg["run_name"] or None,
            config=config,
            job_type="train",
        )
    except Exception as init_exc:  # noqa: BLE001 - turn the raw traceback into guidance
        mo.stop(
            True,
            mo.callout(
                mo.md(
                    f"**Could not start the run.** `{init_exc}`\n\n"
                    f"An `entity ... not found` error means the **W&B entity** is "
                    f"not a team you can write to. Personal-username entities were "
                    f"removed for accounts created after 21 May 2024, so set the "
                    f"**W&B entity** field to one of your teams (find them in the "
                    f"left sidebar at [wandb.ai](https://wandb.ai))."
                ),
                kind="danger",
            ),
        )

    return cfg, collection_name_v, config, registry_name_v, run


@app.cell(hide_code=True)
def _(run):
    # Surface the run link right away so you can watch metrics stream live.
    mo.md(f"**Run started:** [`{run.name}`]({run.url})")
    return


@app.cell
def _(config, run):
    train_ds, test_ds = load_data()
    model, history, final_acc, best_acc = run_training(
        run, config, train_ds, test_ds
    )

    mo.vstack(
        [
            mo.md("### Training summary"),
            mo.ui.table(history, selection=None),
            mo.md(f"**Final test accuracy:** {final_acc:.2%}"),
        ]
    )
    return best_acc, final_acc, model, test_ds, train_ds


@app.cell
def _(
    best_acc,
    cfg,
    collection_name_v,
    config,
    final_acc,
    model,
    registry_name_v,
    run,
    test_ds,
    train_ds,
):
    logged, artifact_name = save_and_log_artifact(
        run,
        model,
        config,
        train_size=len(train_ds),
        test_size=len(test_ds),
        final_acc=final_acc,
        best_acc=best_acc,
    )

    # Link to the Registry unless disabled, capturing the outcome for display
    # rather than crashing the pipeline.
    if not cfg["link_to_registry"]:
        registry_status = {"kind": "disabled"}
    else:
        try:
            registry_status = {
                "kind": "linked",
                "target_path": link_artifact_to_registry(
                    run, logged, registry_name_v, collection_name_v
                ),
            }
        except Exception as link_exc:  # noqa: BLE001 - surface any failure to the reader
            registry_status = {
                "kind": "failed",
                "target_path": f"wandb-registry-{registry_name_v}/{collection_name_v}",
                "error": str(link_exc),
            }

    # Close the run so its summary and any Registry link finalize server-side.
    wandb.finish()
    return artifact_name, registry_status


@app.cell(hide_code=True)
def _(artifact_name, collection_name_v, registry_name_v, registry_status):
    def _registry_callout(status):
        if status["kind"] == "disabled":
            return mo.md(
                "_Registry linking is disabled — the artifact is logged to the run "
                "but not linked to a collection._"
            )
        if status["kind"] == "linked":
            return mo.callout(
                mo.md(
                    f"**Linked to Registry:** `{status['target_path']}` — see "
                    f"[wandb.ai/registry](https://wandb.ai/registry)."
                ),
                kind="success",
            )
        return mo.callout(
            mo.md(
                f"**Registry link failed.** Target `{status['target_path']}` — "
                f"`{status['error']}`\n\n"
                f"- Linking needs at least the **Member** role on the "
                f"Registry. `view-only member cannot write to project` means "
                f"your seat is view-only: the run and artifact succeed, but "
                f"linking is blocked. An admin can grant access from the "
                f"Registry **Members** settings, the Python SDK "
                f"(`wandb.Api().registry(...)` then `add_member()` / "
                f"`update_member()`), or SCIM (`PATCH /scim/Users/{{id}}` with "
                f"`registryRoles`) — see "
                f"https://docs.wandb.ai/guides/registry/configure_registry/. "
                f"Or set **W&B entity** to a team in an org where you have "
                f"Registry write access.\n"
                f"- The Registry `{registry_name_v}` may not exist; an admin "
                f"can create it from the W&B Registry UI.\n"
                f"- On the legacy Model Registry, link with "
                f"`target_path='model-registry/{collection_name_v}'` instead."
            ),
            kind="danger",
        )


    mo.vstack(
        [
            mo.md(f"**Artifact logged:** `{artifact_name}` (alias `latest`)"),
            _registry_callout(registry_status),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Evaluation
    """)
    return


@app.cell
def _(artifact_name, collection_name_v, registry_name_v, run, test_ds):
    api = wandb.Api()
    try:
        consumed = api.artifact(
            f"wandb-registry-{registry_name_v}/{collection_name_v}:latest",
            type="model",
        )
        source = f"registry `wandb-registry-{registry_name_v}/{collection_name_v}:latest`"
    except Exception:  # noqa: BLE001 - registry link may be absent (e.g. a view-only seat)
        consumed = api.artifact(
            f"{run.entity}/{run.project}/{artifact_name}:latest",
            type="model",
        )
        source = f"run artifact `{artifact_name}:latest`"
    weights_dir = consumed.download()

    clf = Net()
    clf.load_state_dict(
        torch.load(f"{weights_dir}/mnist_cnn.pt", map_location="cpu")
    )
    clf.eval()

    rows = []
    n_correct = 0
    with torch.no_grad():
        for i in range(10):
            image, true_label = test_ds[i]
            prediction = clf(image.unsqueeze(0)).argmax(dim=1).item()
            n_correct += int(prediction == true_label)
            # Undo the Normalize transform so the digit renders as a clean image.
            digit = (image * 0.3081 + 0.1307).clamp(0, 1).squeeze().numpy()
            rows.append(
                {
                    "Image": mo.image(digit, width=56, vmin=0, vmax=1),
                    "Label": true_label,
                    "Prediction": prediction,
                }
            )

    mo.vstack(
        [
            mo.md(
                f"**Classify 10 test digits.** Consumed the model from {source}, "
                f"loaded the weights into a fresh network, and ran it on 10 held-out "
                f"MNIST test images — **{n_correct}/10 correct**."
            ),
            mo.ui.table(rows, selection=None),
        ]
    )
    return


@app.cell(hide_code=True)
def _(collection_name_v, registry_name_v, run):
    mo.md(f"""
    ## Verify and next steps

    1. Open the run: [{run.name}]({run.url}) — check the **Training** charts
       and the **System** metrics.
    2. In the run's **Artifacts** tab, confirm `mnist-cnn-{run.id}` is listed
       with its metadata (test accuracy, parameter count, hyperparameters).
    3. At [wandb.ai/registry](https://wandb.ai/registry), open the
       **{registry_name_v.title()}** registry, then the **{collection_name_v}**
       collection, and confirm the linked version.

    **Consume the registered model** from any script or notebook:

    ```python
    import wandb
    art = wandb.Api().artifact(
        "wandb-registry-{registry_name_v}/{collection_name_v}:latest"
    )
    art.download()  # writes mnist_cnn.pt under ./artifacts/
    ```

    **Next steps:** promote a version by adding the `production` alias from
    the Registry UI; re-run with a deeper architecture or a different
    learning rate and compare runs in the W&B UI; or add a W&B Automation to
    trigger evaluation when a new version is linked.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Helper functions
    """)
    return


@app.function
def load_data():
    """Download (or reuse cached) MNIST with the standard normalization."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_ds = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(
        "./data", train=False, download=True, transform=transform
    )
    return train_ds, test_ds


@app.function
def make_loaders(train_ds, test_ds, batch_size):
    """Wrap the datasets in loaders, enabling CUDA niceties when available."""
    loader_kwargs = (
        {"num_workers": 2, "pin_memory": True} if device.type == "cuda" else {}
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs
    )
    test_loader = DataLoader(
        test_ds, batch_size=1000, shuffle=False, **loader_kwargs
    )
    return train_loader, test_loader


@app.function
def train_one_epoch(model, loader, optimizer, epoch, epochs):
    """Run one training epoch, streaming train loss to W&B every 50 steps."""
    model.train()
    for batch_idx, (data, target) in enumerate(
        tqdm(loader, desc=f"epoch {epoch}/{epochs}")
    ):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            wandb.log({"Training/loss": loss.item()})


@app.function
def evaluate(model, loader):
    """Compute test loss and accuracy over a data loader."""
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    n = len(loader.dataset)
    return test_loss / n, correct / n


@app.function
def run_training(run, config, train_ds, test_ds):
    """Train the CNN, logging metrics each epoch; return the model and history."""
    train_loader, test_loader = make_loaders(
        train_ds, test_ds, config["batch_size"]
    )
    model = Net().to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"]
    )

    history = []
    best_acc = 0.0
    test_acc = 0.0
    for epoch in range(1, config["epochs"] + 1):
        train_one_epoch(model, train_loader, optimizer, epoch, config["epochs"])
        test_loss, test_acc = evaluate(model, test_loader)
        best_acc = max(best_acc, test_acc)
        # `train_one_epoch` logs `Training/loss`; logging `Training/accuracy`
        # here keeps both charts in a single "Training" section.
        wandb.log({"Training/accuracy": test_acc})
        history.append(
            {
                "epoch": epoch,
                "test_loss": round(test_loss, 4),
                "test_acc": round(test_acc, 4),
            }
        )
    # Full-precision last-epoch accuracy; `history` rounds only for display.
    return model, history, test_acc, best_acc


@app.function
def save_and_log_artifact(
    run, model, config, train_size, test_size, final_acc, best_acc,
    model_path="mnist_cnn.pt",
):
    """Persist the weights and log them as a `model` Artifact aliased `latest`."""
    torch.save(model.state_dict(), model_path)
    name = f"mnist-cnn-{run.id}"
    artifact = wandb.Artifact(
        name=name,
        type="model",
        description=(
            "Small CNN trained on MNIST. Architecture: 2 conv layers "
            "(10 and 20 filters, 5x5 kernels) + 2 FC layers (50, 10)."
        ),
        metadata={
            "framework": "pytorch",
            "architecture": "CNN",
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "dataset": "MNIST",
            "train_size": train_size,
            "test_size": test_size,
            "test_accuracy": final_acc,
            "best_test_accuracy": best_acc,
            "hyperparameters": dict(config),
        },
    )
    artifact.add_file(model_path)
    logged = run.log_artifact(artifact, aliases=["latest"])
    # Block until the artifact has committed before linking, to avoid a race.
    logged.wait()
    # Return the base name (no version) so callers can build `<name>:latest`.
    return logged, name


@app.function
def link_artifact_to_registry(run, logged, registry_name, collection_name):
    """Link the logged artifact into a Registry collection; return the target."""
    target_path = f"wandb-registry-{registry_name}/{collection_name}"
    run.link_artifact(artifact=logged, target_path=target_path)
    return target_path


if __name__ == "__main__":
    app.run()
