#!/usr/bin/env python3
"""Generate the training dataset: random deals labeled with the exact expected
reward of every hold."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np

from ev import hold_expected_values, rank_payouts
from game import BET, CARDS_PER_HAND, NUM_HOLDS, card_to_code, make_deck
from jacks_or_better import PAYTABLE


def generate_dataset(path: Path, *, hands: int, seed: int) -> None:
    deck = make_deck()
    rng = random.Random(seed)
    payouts = rank_payouts(PAYTABLE)
    cards = np.empty((hands, CARDS_PER_HAND), dtype=np.uint8)
    targets = np.empty((hands, NUM_HOLDS), dtype=np.float32)

    started = time.perf_counter()
    print(f"labeling {hands:,} random deals", flush=True)
    for index in range(hands):
        # Sorted by card code: the same order model.choose_hold feeds the network.
        hand = sorted(rng.sample(deck, CARDS_PER_HAND), key=card_to_code)
        cards[index] = [card_to_code(card) for card in hand]
        targets[index] = np.asarray(hold_expected_values(hand, payouts)) / BET
        if (index + 1) % 100_000 == 0 or index + 1 == hands:
            print(f"labeled {index + 1:,}/{hands:,}", flush=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, cards=cards, targets=targets)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB) in {time.perf_counter() - started:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="where to write the dataset (.npz)")
    parser.add_argument("--hands", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(args.output, hands=args.hands, seed=args.seed)


if __name__ == "__main__":
    main()
