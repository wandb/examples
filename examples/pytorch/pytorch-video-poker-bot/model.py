"""Hold network: encode a hand, score the 32 holds, train, save, and play.

A checkpoint is the learned weights (plus a small config so we can rebuild
the network). Everything runs on CPU — the model is tiny.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from game import BET, Card, VideoPokerGame, deal, make_deck, shuffle_deck

CARDS_PER_HAND = 5
RANK_FEATURES = 13
SUIT_FEATURES = 4
CARD_FEATURES = RANK_FEATURES + SUIT_FEATURES
INPUT_SIZE = CARDS_PER_HAND * CARD_FEATURES
NUM_ACTIONS = 1 << CARDS_PER_HAND  # 32 hold patterns


@dataclass(frozen=True)
class EncodedHand:
    values: np.ndarray
    canonical_cards: tuple[Card, ...]

    def hold_mask(self, action: int, dealt_hand: Sequence[Card]) -> int:
        """Map a canonical action index back onto the physical dealt order."""
        if action < 0 or action >= NUM_ACTIONS:
            raise ValueError(f"action must be 0-{NUM_ACTIONS - 1}")
        held = {
            card
            for index, card in enumerate(self.canonical_cards)
            if action & (1 << index)
        }
        mask = 0
        for index, card in enumerate(dealt_hand):
            if card in held:
                mask |= 1 << index
        if len(held) != bin(mask).count("1"):
            raise ValueError("canonical cards must all be present in dealt hand")
        return mask


def encode_hand(hand: Sequence[Card]) -> EncodedHand:
    """Encode a hand invariant to deal order and physical suit names."""
    if len(hand) != CARDS_PER_HAND:
        raise ValueError("expected exactly 5 cards")
    if len(set(hand)) != CARDS_PER_HAND:
        raise ValueError("hand contains duplicate cards")

    suit_rank_masks = [0] * SUIT_FEATURES
    for card in hand:
        suit_rank_masks[card.suit] |= 1 << (card.rank - 2)
    suit_order = sorted(range(SUIT_FEATURES), key=lambda s: (-suit_rank_masks[s], s))
    suit_map = [0] * SUIT_FEATURES
    for canonical_suit, physical_suit in enumerate(suit_order):
        suit_map[physical_suit] = canonical_suit

    best_cards = tuple(sorted(hand, key=lambda c: (-c.rank, suit_map[c.suit])))
    values = np.zeros((CARDS_PER_HAND, CARD_FEATURES), dtype=np.float32)
    for index, card in enumerate(best_cards):
        values[index, card.rank - 2] = 1.0
        values[index, RANK_FEATURES + suit_map[card.suit]] = 1.0
    return EncodedHand(values=values.reshape(INPUT_SIZE), canonical_cards=best_cards)


class HoldNetwork(nn.Module):
    """MLP that scores each of the 32 possible hold patterns."""

    def __init__(self, hidden_size: int = 256) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = nn.Sequential(
            nn.Linear(INPUT_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, NUM_ACTIONS),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def save_checkpoint(path: Path, *, model: HoldNetwork, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weights": model.state_dict(), "config": config}, path)


def load_checkpoint(path: Path) -> tuple[HoldNetwork, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(payload.get("config", {}))
    # Older checkpoints used "model_state"; prefer "weights".
    state = payload.get("weights") or payload["model_state"]
    model = HoldNetwork(hidden_size=int(config.get("hidden_size", 256)))
    model.load_state_dict(state)
    model.eval()
    return model, config


def choose_hold(model: HoldNetwork, hand: Sequence[Card]) -> int:
    """Pick the highest-scoring hold pattern for this hand."""
    encoded = encode_hand(hand)
    with torch.no_grad():
        scores = model(torch.from_numpy(encoded.values).unsqueeze(0))
        action = int(scores.argmax(dim=1).item())
    return encoded.hold_mask(action, hand)


def play_hands(
    *,
    model: HoldNetwork,
    game: VideoPokerGame,
    hands: int,
    seed: int = 42,
    log_every: int = 10_000,
) -> dict[str, float]:
    """Play greedy hands; return return_pct / profit / wagered / payout."""
    rng = random.Random(seed)
    wagered = 0
    payout = 0
    for hand_num in range(1, hands + 1):
        deck = make_deck()
        shuffle_deck(deck, rng=rng)
        dealt = deal(deck, 5)
        result = game.play_hand_with_state(deck, dealt, choose_hold(model, dealt))
        wagered += BET
        payout += result.payout
        if hand_num % log_every == 0 or hand_num == hands:
            print(
                f"hand {hand_num:,}/{hands:,}  "
                f"return={(payout / wagered) * 100:.2f}%  "
                f"profit={payout - wagered:,}",
                flush=True,
            )
    return {
        "hands": float(hands),
        "bet": float(BET),
        "wagered": float(wagered),
        "payout": float(payout),
        "profit": float(payout - wagered),
        "return_pct": (payout / wagered) * 100,
    }


def train_epoch(
    model: HoldNetwork,
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
    model: HoldNetwork,
    *,
    states: np.ndarray,
    targets: np.ndarray,
    sample_ids: np.ndarray,
    batch_size: int,
) -> dict[str, float]:
    """Loss, % optimal action, and expected return vs the EV labels."""
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
        "mean_ev_regret": total_regret / n,
        "expected_return_pct": (1.0 + total_policy / n) * 100,
    }
