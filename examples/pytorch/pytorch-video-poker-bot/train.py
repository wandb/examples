#!/usr/bin/env python3
"""Train a hold network on exact EV targets and log the run to W&B."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import wandb

from data.dataset import load_dataset
from game import BET
from model import (
    INPUT_SIZE,
    NUM_ACTIONS,
    HoldNetwork,
    save_checkpoint,
    train_epoch,
    validation_metrics,
)

CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "hold-network.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help=".npz from generate_dataset.py")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = load_dataset(args.dataset)
    states = data.states
    train_ids, val_ids, _test_ids = data.split_samples(seed=args.seed)
    model = HoldNetwork(args.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_size": args.hidden_size,
        "seed": args.seed,
        "bet": BET,
        "input_size": INPUT_SIZE,
        "num_actions": NUM_ACTIONS,
        "game": data.metadata.game_id,
        "hands": data.metadata.hand_count,
    }

    # One place for init → log → finish (finish runs automatically on exit).
    with wandb.init(
        project="jacks-or-better",
        name=f"train-h{args.hidden_size}",
        config=config,
    ) as run:
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
                    "val_expected_return_pct": val["expected_return_pct"],
                },
                step=epoch,
            )
            print(
                f"epoch {epoch}/{args.epochs}  "
                f"loss={train_loss:.4f}  "
                f"val_optimal={val['optimal_action_pct']:.1f}%  "
                f"val_return={val['expected_return_pct']:.2f}%"
            )
            if val["mean_ev_regret"] < best_regret:
                best_regret = val["mean_ev_regret"]
                save_checkpoint(args.checkpoint, model=model, config=config)

        artifact = wandb.Artifact(args.checkpoint.stem, type="model")
        artifact.add_file(str(args.checkpoint))
        run.log_artifact(artifact)
        print(f"saved {args.checkpoint}")


if __name__ == "__main__":
    main()
