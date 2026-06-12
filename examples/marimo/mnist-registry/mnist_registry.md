---
title: MNIST -> W&B Registry
marimo-version: 0.23.9
width: medium
header: |-
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
---

```python {.marimo hide_code="true"}
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
    """
)
```

```python {.marimo}
# Imports, device detection, and the input form all live in one cell: this
# is "everything up to collecting your inputs". It defines the form widgets
# but never reads their `.value` — marimo only makes a widget reactive when
# a *different* cell consumes it, which the training cell below does.
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
entity = mo.ui.text(
    value="", label="W&B entity — a team you belong to (blank uses your default)"
)
run_name = mo.ui.text(value="", label="Run name (blank auto-generates)")
api_key = mo.ui.text(
    value="", kind="password", label="W&B API key (blank uses your shell login)"
)

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
        api_key,
        mo.hstack([project, entity, run_name]),
        mo.md("### Registry"),
        mo.hstack([registry_name, collection_name, link_to_registry]),
    ]
)

mo.vstack(
    [
        mo.callout(
            mo.md(f"**Device:** `{device}` &mdash; {device_note}"), kind=device_kind
        ),
        mo.md(
            "## Configure\n\nSet the hyperparameters and W&B targets, then click "
            "**Train model** below. Changing a value here never starts a run on "
            "its own — only the button does."
        ),
        form,
    ]
)
```

```python {.marimo}
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
```

```python {.marimo}
# Everything the Train button triggers, in one cell — no reason to make you
# advance through a chain of output-less code blocks. Each milestone is
# streamed to the cell output with `mo.output.append` as it happens.
mo.stop(
    not train_button.value,
    mo.md(
        "This cell runs the whole pipeline — start the run, train, log "
        "metrics and example predictions, save the model Artifact, and link "
        "it to the Registry. Click **Train model** below to run it."
    ),
)

config = {
    "epochs": epochs.value,
    "batch_size": int(batch_size.value),
    "lr": lr.value,
    "momentum": momentum.value,
    "seed": seed.value,
    "architecture": "CNN",
    "dataset": "MNIST",
}
registry_name_v = registry_name.value.strip()
collection_name_v = collection_name.value.strip()

# Authenticate. Finish any prior run first (marimo keeps the kernel alive
# across re-clicks). A key pasted into the form wins; otherwise fall back to
# ambient login (shell `wandb login`, WANDB_API_KEY, or netrc). The key is
# never written to the run config.
if wandb.run is not None:
    wandb.finish()
if api_key.value:
    wandb.login(key=api_key.value)

torch.manual_seed(config["seed"])

try:
    run = wandb.init(
        project=project.value or None,
        entity=entity.value or None,
        name=run_name.value or None,
        config=config,
        job_type="train",
    )
except Exception as exc:  # noqa: BLE001 - turn the raw traceback into guidance
    mo.stop(
        True,
        mo.callout(
            mo.md(
                f"**Could not start the run.** `{exc}`\n\n"
                f"An `entity ... not found` error means the **W&B entity** is "
                f"not a team you can write to. Personal-username entities were "
                f"removed for accounts created after 21 May 2024, so set the "
                f"**W&B entity** field to one of your teams (find them in the "
                f"left sidebar at [wandb.ai](https://wandb.ai))."
            ),
            kind="danger",
        ),
    )
# Use `epoch` as the x-axis for train/test metrics in the W&B UI.
wandb.define_metric("epoch")
wandb.define_metric("train/*", step_metric="epoch")
wandb.define_metric("test/*", step_metric="epoch")
# Surface the run link right away so you can watch metrics stream live.
mo.output.append(mo.md(f"**Run started:** [`{run.name}`]({run.url})"))

model = Net().to(device)
# `log="gradients"` is the standard choice; `log="all"` also logs parameter
# histograms at extra cost.
wandb.watch(model, log="gradients", log_freq=100)
optimizer = optim.SGD(
    model.parameters(), lr=config["lr"], momentum=config["momentum"]
)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
loader_kwargs = (
    {"num_workers": 2, "pin_memory": True} if device.type == "cuda" else {}
)
train_loader = DataLoader(
    train_ds, batch_size=config["batch_size"], shuffle=True, **loader_kwargs
)
test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False, **loader_kwargs)

history = []
best_acc = 0.0
for epoch in range(1, config["epochs"] + 1):
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
            # Pull up to 16 example predictions from the first batch.
            while len(example_images) < 16 and len(example_images) < data.size(0):
                j = len(example_images)
                example_images.append(
                    wandb.Image(
                        data[j],
                        caption=f"pred={pred[j].item()} true={target[j].item()}",
                    )
                )

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    best_acc = max(best_acc, test_acc)
    wandb.log(
        {
            "test/loss": test_loss,
            "test/accuracy": test_acc,
            "epoch": epoch,
            "examples": example_images,
        }
    )
    history.append(
        {"epoch": epoch, "test_loss": round(test_loss, 4), "test_acc": round(test_acc, 4)}
    )

# Full-precision last-epoch accuracy; `history` rounds only for display.
final_acc = test_acc
mo.output.append(
    mo.vstack(
        [
            mo.md("### Training summary"),
            mo.ui.table(history, selection=None),
            mo.md(f"**Final test accuracy:** {final_acc:.2%}"),
        ]
    )
)

# Save the weights and log them as a model Artifact tagged `latest`.
model_path = "mnist_cnn.pt"
torch.save(model.state_dict(), model_path)
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
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "dataset": "MNIST",
        "train_size": len(train_ds),
        "test_size": len(test_ds),
        "test_accuracy": final_acc,
        "best_test_accuracy": best_acc,
        "hyperparameters": dict(config),
    },
)
artifact.add_file(model_path)
logged = run.log_artifact(artifact, aliases=["latest"])
# Block until the artifact has committed before linking, to avoid a race.
logged.wait()
mo.output.append(mo.md(f"**Artifact logged:** `{artifact.name}` (alias `latest`)"))

# Link to the Registry, surfacing a remediation note instead of crashing.
if link_to_registry.value:
    target_path = f"wandb-registry-{registry_name_v}/{collection_name_v}"
    try:
        run.link_artifact(artifact=logged, target_path=target_path)
        mo.output.append(
            mo.callout(
                mo.md(
                    f"**Linked to Registry:** `{target_path}` — see "
                    f"[wandb.ai/registry](https://wandb.ai/registry)."
                ),
                kind="success",
            )
        )
    except Exception as exc:  # noqa: BLE001 - surface any failure to the reader
        mo.output.append(
            mo.callout(
                mo.md(
                    f"**Registry link failed.** Target `{target_path}` — `{exc}`\n\n"
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
        )
else:
    mo.output.append(
        mo.md(
            "_Registry linking is disabled — the artifact is logged to the run "
            "but not linked to a collection._"
        )
    )

# Close the run so its summary and any Registry link finalize server-side.
wandb.finish()
```

```python {.marimo}
# Placed after the training cell on purpose: it's the explicit "run" trigger
# for the pipeline above. It must be its own cell because that cell reads
# `train_button.value`, and a widget only drives reactivity when a
# *different* cell consumes it. The gate also stops the pipeline from
# running automatically when the notebook opens; run_button's value is True
# only for the cascade a click triggers (then resets to False), so editing
# the form afterwards re-runs the training cell but it stops immediately.
train_button = mo.ui.run_button(label="Train model", kind="success")
mo.vstack(
    [
        train_button,
        mo.md(
            "Runs the training cell above. It is gated so it does not "
            "execute when the notebook opens — click to run, and click "
            "again to retrain after editing the form (the previous run is "
            "finished first)."
        ),
    ]
)
```

```python {.marimo}
# Consume the model: download it from W&B (preferring the registered
# version, falling back to the run's own artifact), load the weights into a
# fresh network, and classify 10 held-out test digits.
api = wandb.Api()
try:
    consumed = api.artifact(
        f"wandb-registry-{registry_name_v}/{collection_name_v}:latest", type="model"
    )
    source = f"registry `wandb-registry-{registry_name_v}/{collection_name_v}:latest`"
except Exception:  # noqa: BLE001 - registry link may be absent (e.g. a view-only seat)
    consumed = api.artifact(
        f"{run.entity}/{run.project}/mnist-cnn-{run.id}:latest", type="model"
    )
    source = f"run artifact `mnist-cnn-{run.id}:latest`"
weights_dir = consumed.download()

clf = Net()
clf.load_state_dict(torch.load(f"{weights_dir}/mnist_cnn.pt", map_location="cpu"))
clf.eval()

cards = []
correct = 0
with torch.no_grad():
    for i in range(10):
        image, true_label = test_ds[i]
        pred = clf(image.unsqueeze(0)).argmax(dim=1).item()
        correct += int(pred == true_label)
        # Undo the Normalize transform so the digit renders as a clean image.
        digit = (image * 0.3081 + 0.1307).clamp(0, 1).squeeze().numpy()
        mark = "✅" if pred == true_label else "❌"
        cards.append(
            mo.vstack(
                [
                    mo.image(digit, width=64, vmin=0, vmax=1),
                    mo.md(f"{mark} **{pred}** · true {true_label}"),
                ],
                align="center",
            )
        )

mo.vstack(
    [
        mo.md(
            f"## Classify 10 test digits\n\nConsumed the model from {source}, "
            f"loaded the weights into a fresh network, and ran it on 10 held-out "
            f"MNIST test images — **{correct}/10 correct**."
        ),
        mo.hstack(cards, wrap=True, justify="start"),
    ]
)
```

````python {.marimo hide_code="true"}
# Renders only after a run exists (it consumes `run` from the training
# cell), so it appears once training finishes.
mo.md(
    f"""
    ## Verify and next steps

    1. Open the run: [{run.name}]({run.url}) — check the **Charts**,
       **System**, and **Examples** panels.
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
    """
)
````