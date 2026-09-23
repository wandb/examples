#!/usr/bin/env python3
"""Fetch a trained network from W&B and play hands with it, logging as it goes."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import wandb

from game import BET, CARDS_PER_HAND, VideoPokerGame, deal, make_deck, shuffle_deck
from jacks_or_better import make_game
from model import Network, choose_hold, load_checkpoint


def play_hands(
    run: wandb.Run,
    model: Network,
    game: VideoPokerGame,
    *,
    hands: int,
    seed: int,
    log_every: int = 100,
) -> None:
    """Play greedy hands, logging and printing running totals every `log_every` hands."""
    rng = random.Random(seed)
    wagered = 0
    payout = 0
    for hand_num in range(1, hands + 1):
        deck = make_deck()
        shuffle_deck(deck, rng=rng)
        dealt = deal(deck, CARDS_PER_HAND)
        payout += game.play_hand(deck, dealt, choose_hold(model, dealt))
        wagered += BET
        if hand_num % log_every == 0 or hand_num == hands:
            return_pct = (payout / wagered) * 100
            run.log(
                {
                    "hands_played": hand_num,
                    "wagered": wagered,
                    "payout": payout,
                    "profit": payout - wagered,
                    "return_pct": return_pct,
                },
                step=hand_num,
            )
            print(f"hand {hand_num:,}/{hands:,}  return={return_pct:.2f}%  profit={payout - wagered:,}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, help="model artifact from train.py, e.g. my-model:latest")
    parser.add_argument("--project", required=True, help="W&B project to log to")
    parser.add_argument("--run-name", required=True, help="name for this W&B run")
    parser.add_argument("--hands", type=int, default=100_000)
    parser.add_argument("--seed", type=int, help="default: random; the seed used is logged to W&B")
    args = parser.parse_args()
    if args.seed is None:
        args.seed = random.randrange(2**32)

    with wandb.init(
        project=args.project,
        name=args.run_name,
        job_type="evaluation",
        group="eval",
        config={"artifact": args.artifact, "hands": args.hands, "seed": args.seed},
    ) as run:
        # use_artifact records that this run consumed the model, linking it to the training run.
        checkpoint = Path(run.use_artifact(args.artifact).file())
        model = load_checkpoint(checkpoint)
        play_hands(run, model, make_game(), hands=args.hands, seed=args.seed)


if __name__ == "__main__":
    main()
