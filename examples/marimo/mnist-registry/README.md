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
- A W&B account. Run `wandb login` once in your shell before launching the
  notebook &mdash; this notebook does not prompt for an API key interactively.
- A W&B **Registry** must exist in your org for the final linking step. The
  built-in Model registry is provisioned automatically in newer orgs. If
  linking fails, the notebook surfaces a remediation message in the last
  Registry cell instead of crashing.
- GPU is optional. Defaults are tuned to finish in roughly two minutes on CPU.

## Run

The recommended entry point is `uvx` with marimo's sandbox mode &mdash; it
creates an isolated venv from the inline dependencies in the notebook:

```bash
uvx marimo edit mnist_registry.py --sandbox
```

Marimo opens in your browser. Adjust hyperparameters in the form, then click
**Train model** to start the run. The run URL appears inline as soon as
training begins.

If you prefer pip:

```bash
pip install -r requirements.txt
marimo edit mnist_registry.py
```

The notebook is interactive-only by design: training is gated by a button
click, so `marimo run` will render the form but never start training without
an explicit click.

## What you get

After a successful run:

- A W&B run with training and test metrics, gradient histograms (`wandb.watch`),
  and up to 16 example test-set predictions logged as images.
- A model Artifact named `mnist-cnn-<run-id>` of type `model` with metadata
  for test accuracy, parameter count, dataset sizes, and the full
  hyperparameter dict. Tagged with the `latest` alias.
- A version of that Artifact linked into the configured Registry collection
  (default: `wandb-registry-model/MNIST Classifiers`).

To consume the registered model from another script or notebook:

```python
import wandb
api = wandb.Api()
art = api.artifact("wandb-registry-model/MNIST Classifiers:latest")
art.download()  # writes mnist_cnn.pt under ./artifacts/
```

## Design notes

- **Training is gated by a button.** Marimo cells re-run reactively when their
  inputs change. Before the first click of **Train model**, slider changes do
  not start a run. After a run has completed, clicking **Train model** again
  starts a new run with whatever the form values are at that moment; the
  previous run is finished cleanly first.
- **`wandb.run` is finished defensively** at the top of the training cell so
  the second click of **Train model** does not nest runs in the same marimo
  kernel.
- **`logged.wait()` is called** after `log_artifact` and before
  `link_artifact` to avoid a race where the link tries to resolve a version
  that has not finished committing server-side.
- **Registry failures soft-fail.** If `link_artifact` raises &mdash; usually
  because the Registry does not exist in your org &mdash; the notebook
  surfaces remediation guidance via `mo.callout` rather than aborting.

## Reference

The CNN architecture and training loop mirror
[`examples/pytorch/pytorch-cnn-mnist/main.py`](../../pytorch/pytorch-cnn-mnist/main.py).
The Registry linking pattern follows
[`colabs/wandb_registry/zoo_wandb.ipynb`](../../../colabs/wandb_registry/zoo_wandb.ipynb).
