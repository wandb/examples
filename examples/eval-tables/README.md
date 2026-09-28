# Eval Tables

> [!NOTE]
> Eval Tables is currently in **Public Preview**. Functionality and availability
> are subject to change.

For an overview of the feature, see the
[Eval Tables Documentation](https://docs.wandb.ai/models/evaltables).

`eval_tables_demo.py` trains a small MNIST classifier with three configs, one
W&B run per config. After each epoch, each run logs a `wandb.EvalTable` of
predictions on the same validation images. The MNIST data, model and training
helpers are in `mnist.py`.

| Role | Columns |
| --- | --- |
| Input | `image` (`wandb.Image`), `label` |
| Output | `pred`, `confidence` |
| Score | `correct`, `loss` |

Each run also logs `epoch`, `train_loss`, `val_loss` and `val_accuracy`.

## Setup

```sh
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run

```sh
wandb login
python eval_tables_demo.py
```

The script logs to the `eval-tables-demo` project. Pass `--entity`, `--project`,
`--epochs` or `--val-rows` to change the defaults, and see `--help` for the
rest. MNIST downloads to `./MNIST` on the first run.
