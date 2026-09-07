# Terradev + Weights & Biases

This example shows how to combine [Terradev](https://github.com/theoddden/terradev)'s cross-cloud GPU cost optimization with Weights & Biases experiment tracking.

## What it does

1. **Find the cheapest GPU** for your workload using Terradev's multi-cloud quote engine.
2. **Log infrastructure metadata** (provider, region, cost per hour) to a W&B run via `wandb.config`.
3. **Track training metrics** (loss, accuracy, GPU utilization, cumulative cost) in W&B.

By default the script runs in **demo mode** with a sample quote, so you can try it without any cloud credentials. Pass `--live` to fetch real pricing from Terradev.

## Setup

```bash
cd examples/terradev
pip install -r requirements.txt
```

For live quotes you also need Terradev:

```bash
pip install terradev-cli
terradev configure --provider runpod  # or any supported provider
```

Set your W&B credentials:

```bash
wandb login
# or
export WANDB_API_KEY=...
```

## Run

### Demo mode (no cloud credentials needed)

```bash
python terradev_wandb_example.py
```

### Live quote mode

```bash
python terradev_wandb_example.py --live --gpu-type H100
```

You will see output similar to:

```
Best live quote: runpod us-east-1 at $1.25/hr
W&B run started: https://wandb.ai/<entity>/terradev-wandb-example/runs/<run-id>
Run complete. View it in your W&B project.
```

## Files

- `terradev_wandb_example.py` — main example script.
- `requirements.txt` — Python dependencies.

## Next steps

- Provision the quoted instance with `terradev provision -g <gpu-type> --providers <provider>`.
- Use this script as a starting point for your own cost-aware training jobs.
