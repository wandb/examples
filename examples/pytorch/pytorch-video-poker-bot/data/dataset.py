"""Load the Jacks or Better training dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from model import (
    CARD_FEATURES,
    CARDS_PER_HAND,
    INPUT_SIZE,
    NUM_ACTIONS,
    RANK_FEATURES,
)


@dataclass(frozen=True)
class DatasetMeta:
    game_id: str
    bet: int
    hand_count: int


@dataclass(frozen=True)
class HoldDataset:
    metadata: DatasetMeta
    cards: np.ndarray
    targets: np.ndarray

    @property
    def states(self) -> np.ndarray:
        return encode_cards(self.cards)

    def split_samples(
        self,
        *,
        seed: int,
        train_fraction: float = 0.8,
        validation_fraction: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Shuffle into train / validation / test (80/10/10)."""
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(self.cards))
        train_end = int(len(order) * train_fraction)
        val_end = train_end + int(len(order) * validation_fraction)
        return order[:train_end], order[train_end:val_end], order[val_end:]


def encode_cards(cards: np.ndarray) -> np.ndarray:
    """Batch-encode stored card codes into model inputs."""
    card_codes = np.asarray(cards)
    if card_codes.ndim != 2 or card_codes.shape[1] != CARDS_PER_HAND:
        raise ValueError("cards must have shape (N, 5)")
    ranks = card_codes // 4
    suits = card_codes % 4
    rows = np.arange(len(card_codes))[:, None]
    positions = np.arange(CARDS_PER_HAND)[None, :]
    encoded = np.zeros((len(card_codes), CARDS_PER_HAND, CARD_FEATURES), dtype=np.float32)
    encoded[rows, positions, ranks] = 1.0
    encoded[rows, positions, RANK_FEATURES + suits] = 1.0
    return encoded.reshape(len(card_codes), INPUT_SIZE)


def load_dataset(path: Path) -> HoldDataset:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; create it with generate_dataset.py {path}")
    with np.load(path, allow_pickle=False) as payload:
        meta = _read_meta(payload)
        cards = payload["cards"]
        targets = payload["targets"]

    if cards.shape != (meta.hand_count, CARDS_PER_HAND):
        raise ValueError(f"invalid cards shape: {cards.shape}")
    if targets.shape != (meta.hand_count, NUM_ACTIONS):
        raise ValueError(f"invalid targets shape: {targets.shape}")

    return HoldDataset(metadata=meta, cards=cards, targets=targets)


def _read_meta(payload: np.lib.npyio.NpzFile) -> DatasetMeta:
    return DatasetMeta(
        game_id=str(payload["game"][0]),
        bet=int(payload["bet"][0]),
        hand_count=int(payload["hand_count"][0]),
    )
