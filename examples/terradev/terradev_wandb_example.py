#!/usr/bin/env python3
"""
Terradev + Weights & Biases example integration.

This example shows how to use Terradev's cross-cloud GPU pricing to
choose the cheapest provider for a workload, then log that infrastructure
metadata and training metrics to a W&B run.

By default the script runs in demo mode with a sample quote. Set `--live` to
fetch a real quote from the Terradev CLI (requires a configured Terradev
installation and cloud credentials).

Usage:
    python terradev_wandb_example.py
    python terradev_wandb_example.py --live --gpu-type H100
"""

import argparse
import os
import re
import subprocess
from typing import Any, Dict, Optional

import wandb


SAMPLE_QUOTE = {
    "provider": "runpod",
    "region": "us-east-1",
    "price": 1.99,
    "gpu_type": "A100",
    "instance_type": "a100-80gb",
    "gpu_count": 1,
}


def get_terradev_quote(gpu_type: str) -> Optional[Dict[str, Any]]:
    """Call `terradev quote` and parse the best quote line.

    Terradev's quote output includes a line like:

        Best: $1.25/hr on runpod (us-east-1)

    We extract the price, provider, and region from that line. If Terradev is
    not installed or the command fails, we fall back to the sample quote.
    """
    try:
        result = subprocess.run(
            ["terradev", "quote", "-g", gpu_type],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except FileNotFoundError:
        print("Terradev CLI not found. Install it (`pip install terradev-cli`) to use --live.")
        return None

    if result.returncode != 0:
        print("Terradev quote failed. Using sample quote for demo.")
        return None

    best_line = next(
        (line for line in result.stdout.splitlines() if line.startswith("Best:")),
        None,
    )
    if not best_line:
        return None

    best_match = re.search(
        r"Best: \$([\d.]+)/hr on ([^(]+) \(([^)]+)\)", best_line
    )
    if not best_match:
        return None

    return {
        "provider": best_match.group(2).strip(),
        "region": best_match.group(3).strip(),
        "price": float(best_match.group(1)),
        "gpu_type": gpu_type,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Log Terradev GPU pricing and training metrics to W&B"
    )
    parser.add_argument(
        "--gpu-type",
        default="A100",
        help="GPU type to quote with Terradev (default: A100)",
    )
    parser.add_argument(
        "--project",
        default="terradev-wandb-example",
        help="W&B project name",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Fetch a live quote from the Terradev CLI",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of simulated training steps",
    )
    args = parser.parse_args()

    quote = None
    if args.live:
        quote = get_terradev_quote(args.gpu_type)

    if not quote:
        quote = SAMPLE_QUOTE.copy()
        quote["gpu_type"] = args.gpu_type
        print(f"Using sample quote: {quote['provider']} {quote['region']} at ${quote['price']}/hr")
    else:
        print(
            f"Best live quote: {quote['provider']} {quote['region']} at ${quote['price']}/hr"
        )

    # Enrich with sample instance details when they are missing.
    quote.setdefault("instance_type", quote.get("gpu_type", args.gpu_type))
    quote.setdefault("gpu_count", 1)

    # Initialize a W&B run with the selected infrastructure as config.
    run = wandb.init(
        project=args.project,
        config={
            "gpu_type": quote["gpu_type"],
            "provider": quote["provider"],
            "region": quote["region"],
            "cost_per_hour": quote["price"],
            "instance_type": quote["instance_type"],
            "gpu_count": quote["gpu_count"],
        },
    )

    print(f"W&B run started: {run.url}")

    # Simulated training loop. In a real workload this would be replaced by
    # actual model training on the Terradev-provisioned instance.
    for step in range(args.steps):
        loss = 1.0 / (step + 1) ** 0.5
        accuracy = 1.0 - loss
        gpu_utilization = 70.0 + 20.0 * ((step % 10) / 10.0)

        # Estimate cumulative cost assuming one step takes ~1 second.
        cumulative_cost = quote["price"] * (step / 3600.0)

        wandb.log(
            {
                "step": step,
                "loss": loss,
                "accuracy": accuracy,
                "gpu_utilization": gpu_utilization,
                "cost_per_hour": quote["price"],
                "cumulative_cost": cumulative_cost,
            }
        )

    wandb.finish()
    print("Run complete. View it in your W&B project.")


if __name__ == "__main__":
    main()
