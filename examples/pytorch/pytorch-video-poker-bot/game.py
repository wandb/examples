"""Video poker core: cards, ranks, paytable, and the deal/hold/draw loop.

The bet is always five credits, baked into BET below.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping, Sequence

BET = 5  # credits bet per hand
CARDS_PER_HAND = 5
DECK_SIZE = 52
NUM_HOLDS = 1 << CARDS_PER_HAND  # 32 ways to choose which cards to hold

RANKS = tuple(range(2, 15))  # 2..14, Ace high
SUITS = (0, 1, 2, 3)
RANK_CHARS = {11: "J", 12: "Q", 13: "K", 14: "A"}
SUIT_CHARS = ("s", "h", "d", "c")


@dataclass(frozen=True)
class Card:
    rank: int
    suit: int

    def __post_init__(self) -> None:
        if self.rank not in RANKS:
            raise ValueError(f"invalid rank: {self.rank}")
        if self.suit not in SUITS:
            raise ValueError(f"invalid suit: {self.suit}")

    def __str__(self) -> str:
        rank = str(self.rank) if self.rank <= 10 else RANK_CHARS[self.rank]
        return f"{rank}{SUIT_CHARS[self.suit]}"


def make_deck() -> list[Card]:
    return [Card(rank=rank, suit=suit) for suit in SUITS for rank in RANKS]


def card_to_code(card: Card) -> int:
    """Pack a card into 0..51 (rank-major)."""
    return (card.rank - 2) * 4 + card.suit


def hand_to_codes(hand: Sequence[Card]) -> list[int]:
    return [card_to_code(card) for card in hand]


def shuffle_deck(deck: list[Card], rng: random.Random | None = None) -> None:
    (rng or random).shuffle(deck)


def deal(deck: list[Card], count: int) -> list[Card]:
    if count > len(deck):
        raise ValueError("not enough cards left in deck")
    dealt = deck[:count]
    del deck[:count]
    return dealt


def apply_hold(hand: Sequence[Card], hold_mask: int, deck: list[Card]) -> list[Card]:
    if not 0 <= hold_mask < NUM_HOLDS:
        raise ValueError(f"hold_mask must be 0-{NUM_HOLDS - 1}")
    held = [card for index, card in enumerate(hand) if hold_mask & (1 << index)]
    draw_count = CARDS_PER_HAND - len(held)
    return held + (deal(deck, draw_count) if draw_count else [])


class HandRank(Enum):
    ROYAL_FLUSH = "royal_flush"
    STRAIGHT_FLUSH = "straight_flush"
    FOUR_OF_A_KIND = "four_of_a_kind"
    FULL_HOUSE = "full_house"
    FLUSH = "flush"
    STRAIGHT = "straight"
    THREE_OF_A_KIND = "three_of_a_kind"
    TWO_PAIR = "two_pair"
    JACKS_OR_BETTER = "jacks_or_better"
    NOTHING = "nothing"


@dataclass(frozen=True)
class EvaluatedHand:
    rank: HandRank


@dataclass(frozen=True)
class Paytable:
    """Maps a hand rank to total credits paid (for a fixed BET)."""

    payouts: Mapping[HandRank, int] = field(default_factory=dict)

    def payout_for(self, rank: HandRank) -> int:
        return self.payouts.get(rank, 0)


@dataclass
class VideoPokerGame:
    paytable: Paytable
    evaluate: Callable[[Sequence[Card]], EvaluatedHand]

    def play_hand(self, deck: list[Card], hand: Sequence[Card], hold_mask: int) -> int:
        """Hold, draw from `deck`, and return the reward in credits."""
        if len(hand) != CARDS_PER_HAND:
            raise ValueError("expected a 5-card hand")
        final = apply_hold(hand, hold_mask, deck)
        return self.paytable.payout_for(self.evaluate(final).rank)
