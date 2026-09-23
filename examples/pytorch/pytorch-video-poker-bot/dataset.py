"""Load a dataset written by generate_dataset.py."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from game import CARDS_PER_HAND, NUM_HOLDS
from model import encode_cards


@dataclass(frozen=True)
class HoldDataset:
    cards: np.ndarray  # (N, 5) card codes, sorted
    targets: np.ndarray  # (N, 32) expected profit per credit bet for each hold

    @property
    def states(self) -> np.ndarray:
        return encode_cards(self.cards)

    def split_samples(
        self,
        *,
        seed: int,
        validation_fraction: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Shuffle into train / validation (90/10)."""
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(self.cards))
        val_start = len(order) - int(len(order) * validation_fraction)
        return order[:val_start], order[val_start:]


def load_dataset(path: Path) -> HoldDataset:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; create it with generate_dataset.py --output {path}")
    with np.load(path, allow_pickle=False) as payload:
        cards = payload["cards"]
        targets = payload["targets"]

    if cards.ndim != 2 or cards.shape[1] != CARDS_PER_HAND:
        raise ValueError(f"invalid cards shape: {cards.shape}")
    if targets.shape != (len(cards), NUM_HOLDS):
        raise ValueError(f"invalid targets shape: {targets.shape}")

    return HoldDataset(cards=cards, targets=targets)
