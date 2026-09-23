#!/usr/bin/env python3
"""Evaluate a trained hold-network checkpoint with greedy play."""

from __future__ import annotations

import argparse
from pathlib import Path

import wandb

from jacks_or_better import make_game
from model import load_checkpoint, play_hands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help=".pt saved by train.py")
    parser.add_argument("--project", required=True, help="W&B project to log to")
    parser.add_argument("--run-name", required=True, help="name for this W&B run")
    parser.add_argument("--hands", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    model, config = load_checkpoint(args.checkpoint)
    game = make_game()

    print(f"checkpoint={args.checkpoint} hands={args.hands:,}", flush=True)

    with wandb.init(
        project=args.project,
        name=args.run_name,
        job_type="evaluation",
        config={
            "checkpoint": str(args.checkpoint),
            "hands": args.hands,
            "seed": args.seed,
            **config,
        },
    ) as run:
        metrics = play_hands(
            model=model,
            game=game,
            hands=args.hands,
            seed=args.seed,
        )
        run.summary.update(metrics)
        run.log(metrics)
        print(f"return_pct={metrics['return_pct']:.2f}%  profit={int(metrics['profit']):,}")


if __name__ == "__main__":
    main()
