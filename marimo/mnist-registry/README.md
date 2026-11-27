# MNIST -> W&B Registry (marimo)

A [marimo](https://marimo.io) notebook that trains a small CNN on MNIST with
PyTorch, tracks the run in Weights & Biases, saves the trained model as a W&B
Artifact, and links that Artifact to a collection in the **W&B Registry**.

The notebook is the first marimo example in this repo and is intentionally
self-contained: dependencies are declared in a [PEP 723](https://peps.python.org/pep-0723/)
inline-script block at the top of `mnist_registry.py`, so [`uv`](https://docs.astral.sh/uv/)
can resolve them automatically.

## Prerequisites

- Python 3.10 or newer.
- A W&B account, authenticated one of two ways: run `wandb login` in your
  shell before launching the notebook, or paste your key into the **W&B API
  key** field in the form. Get your key from
  [wandb.ai/authorize](https://wandb.ai/authorize).
- A W&B **team** to write the run to, set in the **W&B entity** field. Accounts
  created after May 2024 have no personal entity, so the run must go to a team
  — your username will not work as an entity.
- A W&B **Registry** must exist in your org, and your account needs at least
  the **Member** role on it for the final linking step (linking an artifact is
  a write action). The built-in Model registry is provisioned automatically in
  newer orgs. If linking fails (for example, from a view-only seat), the
  notebook surfaces a remediation message in the last Registry cell instead of
  crashing. See
  [configuring registry access](https://docs.wandb.ai/guides/registry/configure_registry/).
- GPU is optional. Defaults are tuned to finish in roughly two minutes on CPU.

## Run

Use `uvx` with marimo's sandbox mode &mdash; it creates an isolated virtual
environment from the inline dependencies in the notebook:

```bash
uvx marimo edit mnist_registry.py --sandbox
```

marimo opens in your browser. Adjust hyperparameters in the form, then click
**Train model** to start the run. The run URL appears inline as soon as
training begins.

If you prefer pip:

```bash
pip install -r requirements.txt
marimo edit mnist_registry.py
```

The notebook is interactive-only by design: training is gated by submitting
the form, so `marimo run` renders the form but never starts training until you
click **Train model**.

## What you get

After a successful run:

- A W&B run whose **Training** section charts loss and accuracy, alongside a
  confusion matrix, a per-digit PR curve, a table of example predictions (with
  images), and penultimate-layer embeddings you can open in a 2D projector —
  one cluster per digit. Final/best accuracy and the parameter count are
  surfaced on the run overview.
- A model Artifact named `mnist-cnn-<run-id>` of type `model` with metadata
  for test accuracy, parameter count, dataset sizes, and the full
  hyperparameter dict. Tagged with the `latest` alias.
- A version of that Artifact linked into the configured Registry collection
  (default: `wandb-registry-model/MNIST Classifiers`).

The notebook then **consumes the model in place**: its Evaluation cell
downloads the artifact (preferring the registered version, falling back to the
run's own artifact) and classifies ten held-out test digits, showing predicted
versus true labels. To consume it from another script or notebook:

```python
import wandb
api = wandb.Api()
art = api.artifact("wandb-registry-model/MNIST Classifiers:latest")
art.download()  # writes mnist_cnn.pt under ./artifacts/
```

## Design notes

- **Training is gated by a form.** Hyperparameters live in a marimo form, so
  changing a field does nothing until you submit it with **Train model**.
  Submitting again after a run starts a new run with the current values; the
  previous run is finished cleanly first.
- **`wandb.run` finishes defensively** at the top of the training cell so
  a second click of **Train model** does not nest runs in the same marimo
  kernel.
- **`logged.wait()` runs** after `log_artifact` and before `link_artifact`
  to avoid a race where the link tries to resolve a version that has not
  finished committing server-side.
- **Registry failures soft-fail.** If linking raises &mdash; usually a
  view-only seat or a Registry that does not exist in your org &mdash; the
  notebook surfaces remediation guidance through `mo.callout` rather than
  aborting; the run and artifact still succeed.

## Reference

The CNN architecture and training loop mirror
[`examples/pytorch/pytorch-cnn-mnist/main.py`](../../pytorch/pytorch-cnn-mnist/main.py).
The Registry linking pattern follows
[`colabs/wandb_registry/zoo_wandb.ipynb`](../../../colabs/wandb_registry/zoo_wandb.ipynb).
