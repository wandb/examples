"""Build the Jacks or Better training dataset (1M random deals, EV-labeled)."""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np

from ev import hold_expected_values
from game import BET, card_to_code, make_deck
from jacks_or_better import GAME_ID, PAYTABLE
from model import NUM_ACTIONS, encode_hand

DEFAULT_HANDS = 1_000_000
DEFAULT_SEED = 42


def generate_dataset(
    path: Path,
    *,
    hands: int = DEFAULT_HANDS,
    seed: int = DEFAULT_SEED,
    force: bool = False,
) -> Path:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}; pass --force")

    deck = make_deck()
    rng = random.Random(seed)
    cards = np.empty((hands, 5), dtype=np.uint8)
    targets = np.empty((hands, NUM_ACTIONS), dtype=np.float32)

    started = time.perf_counter()
    print(f"labeling {hands:,} random deals", flush=True)
    for index in range(hands):
        encoded = encode_hand(rng.sample(deck, 5))
        cards[index] = [card_to_code(card) for card in encoded.canonical_cards]
        values = np.asarray(
            hold_expected_values(encoded.canonical_cards, PAYTABLE),
            dtype=np.float32,
        )
        targets[index] = values / BET
        if (index + 1) % 100_000 == 0 or index + 1 == hands:
            print(f"labeled {index + 1:,}/{hands:,}", flush=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cards=cards,
        targets=targets,
        game=np.asarray([GAME_ID]),
        bet=np.asarray([BET], dtype=np.uint8),
        hand_count=np.asarray([hands], dtype=np.uint32),
        seed=np.asarray([seed], dtype=np.uint64),
    )
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB) in {time.perf_counter() - started:.1f}s")
    return path
