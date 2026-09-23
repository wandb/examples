#!/usr/bin/env python3
"""Train the network on exact expected-reward targets and log the run to W&B."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import wandb

from dataset import load_dataset
from model import Network, save_checkpoint, train_epoch, validation_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help=".npz from generate_dataset.py")
    parser.add_argument("--checkpoint", type=Path, required=True, help="where to save the trained model (.pt)")
    parser.add_argument("--artifact-name", required=True, help="W&B model artifact to upload the checkpoint as")
    parser.add_argument("--project", required=True, help="W&B project to log to")
    parser.add_argument("--run-name", required=True, help="name for this W&B run")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    data = load_dataset(args.dataset)
    states = data.states
    train_ids, val_ids = data.split_samples(seed=args.seed)
    model = Network(args.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_size": args.hidden_size,
        "seed": args.seed,
        "hands": len(data.cards),
    }

    # One place for init → log → finish (finish runs automatically on exit).
    with wandb.init(project=args.project, name=args.run_name, config=config) as run:
        best_regret = float("inf")
        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(
                model,
                optimizer,
                states=states,
                targets=data.targets,
                train_ids=train_ids,
                batch_size=args.batch_size,
                rng=rng,
            )
            val = validation_metrics(
                model,
                states=states,
                targets=data.targets,
                sample_ids=val_ids,
                batch_size=args.batch_size,
            )
            run.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val["loss"],
                    "val_optimal_action_pct": val["optimal_action_pct"],
                    "val_mean_regret": val["mean_regret"],
                    "val_expected_return_pct": val["expected_return_pct"],
                },
                step=epoch,
            )
            print(
                f"epoch {epoch}/{args.epochs}  "
                f"loss={train_loss:.4f}  "
                f"val_optimal={val['optimal_action_pct']:.1f}%  "
                f"val_regret={val['mean_regret']:.4f}  "
                f"val_return={val['expected_return_pct']:.2f}%"
            )
            if val["mean_regret"] < best_regret:
                best_regret = val["mean_regret"]
                save_checkpoint(args.checkpoint, model)

        artifact = wandb.Artifact(args.artifact_name, type="model")
        artifact.add_file(str(args.checkpoint))
        run.log_artifact(artifact)
        print(f"saved {args.checkpoint} and logged artifact {args.artifact_name}")


if __name__ == "__main__":
    main()
