"""The network: encode hands, score the 32 holds, train, validate, save and load.

Everything runs on CPU; the model is tiny.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from game import CARDS_PER_HAND, NUM_HOLDS, Card, card_to_code

RANK_FEATURES = 13
SUIT_FEATURES = 4
CARD_FEATURES = RANK_FEATURES + SUIT_FEATURES
INPUT_SIZE = CARDS_PER_HAND * CARD_FEATURES


def encode_cards(cards: np.ndarray) -> np.ndarray:
    """One-hot encode (N, 5) card codes into (N, INPUT_SIZE) network inputs."""
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


class Network(nn.Module):
    """MLP that scores each of the 32 possible hold patterns."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = nn.Sequential(
            nn.Linear(INPUT_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, NUM_HOLDS),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def save_checkpoint(path: Path, model: Network) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"hidden_size": model.hidden_size, "weights": model.state_dict()}, path)


def load_checkpoint(path: Path) -> Network:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = Network(payload["hidden_size"])
    model.load_state_dict(payload["weights"])
    model.eval()
    return model


def choose_hold(model: Network, hand: Sequence[Card]) -> int:
    """Pick the highest-scoring hold; returns a hold mask over `hand` as dealt.

    The network sees cards sorted by card code, the same order the dataset uses.
    """
    order = sorted(range(CARDS_PER_HAND), key=lambda i: card_to_code(hand[i]))
    codes = np.array([[card_to_code(hand[i]) for i in order]])
    with torch.no_grad():
        action = int(model(torch.from_numpy(encode_cards(codes))).argmax(dim=1).item())
    return sum(1 << order[bit] for bit in range(CARDS_PER_HAND) if action & (1 << bit))


def train_epoch(
    model: Network,
    optimizer: torch.optim.Optimizer,
    *,
    states: np.ndarray,
    targets: np.ndarray,
    train_ids: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> float:
    model.train()
    order = rng.permutation(len(train_ids))
    total_loss = 0.0
    n = 0
    for start in range(0, len(order), batch_size):
        ids = train_ids[order[start : start + batch_size]]
        loss = F.smooth_l1_loss(
            model(torch.from_numpy(states[ids])),
            torch.from_numpy(targets[ids]),
        )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        total_loss += float(loss.item()) * len(ids)
        n += len(ids)
    return total_loss / n


@torch.no_grad()
def validation_metrics(
    model: Network,
    *,
    states: np.ndarray,
    targets: np.ndarray,
    sample_ids: np.ndarray,
    batch_size: int,
) -> dict[str, float]:
    """Loss, % optimal holds, mean regret, and expected return vs the EV labels."""
    model.eval()
    total_loss = total_regret = total_policy = 0.0
    optimal = n = 0
    for start in range(0, len(sample_ids), batch_size):
        ids = sample_ids[start : start + batch_size]
        batch = targets[ids]
        preds = model(torch.from_numpy(states[ids]))
        total_loss += float(
            F.smooth_l1_loss(
                preds, torch.from_numpy(batch), reduction="sum"
            ).item()
        )
        chosen = preds.argmax(dim=1).numpy()
        best = batch.max(axis=1)
        chosen_values = batch[np.arange(len(ids)), chosen]
        regrets = best - chosen_values
        total_regret += float(regrets.sum())
        total_policy += float(chosen_values.sum())
        optimal += int(np.count_nonzero(regrets <= 1e-7))
        n += len(ids)
    return {
        "loss": total_loss / (n * targets.shape[1]),
        "optimal_action_pct": (optimal / n) * 100,
        "mean_regret": total_regret / n,
        "expected_return_pct": (1.0 + total_policy / n) * 100,
    }
