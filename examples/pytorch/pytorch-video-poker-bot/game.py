"""Video poker core: cards, ranks, paytable, and the deal/hold/draw loop.

The bet is always five coins — baked into BET below.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping, Sequence

BET = 5  # always max-coin Jacks or Better

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


def parse_card(text: str) -> Card:
    """Parse strings like 'Ah', 'Td', '7s'."""
    text = text.strip()
    if len(text) != 2:
        raise ValueError(f"invalid card: {text}")
    rank_char, suit_char = text[0].upper(), text[1].lower()
    rank_map = {"T": 10, **{v: k for k, v in RANK_CHARS.items()}}
    if rank_char.isdigit():
        rank = int(rank_char)
    else:
        rank = rank_map.get(rank_char)
    if rank is None or rank not in RANKS:
        raise ValueError(f"invalid rank in card: {text}")
    try:
        suit = SUIT_CHARS.index(suit_char)
    except ValueError as exc:
        raise ValueError(f"invalid suit in card: {text}") from exc
    return Card(rank=rank, suit=suit)


def cards_from_strings(values: Sequence[str]) -> list[Card]:
    return [parse_card(value) for value in values]


def make_deck() -> list[Card]:
    return [Card(rank=rank, suit=suit) for suit in SUITS for rank in RANKS]


def card_to_code(card: Card) -> int:
    """Pack a card into 0..51 (rank-major)."""
    return (card.rank - 2) * 4 + card.suit


def code_to_card(code: int) -> Card:
    return Card(rank=(code // 4) + 2, suit=code % 4)


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
    if hold_mask < 0 or hold_mask > 31:
        raise ValueError("hold_mask must be a 5-bit integer (0-31)")
    held = [card for index, card in enumerate(hand) if hold_mask & (1 << index)]
    draw_count = 5 - len(held)
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

    def __str__(self) -> str:
        return self.rank.value


@dataclass(frozen=True)
class Paytable:
    """Maps a hand rank to total credits paid (for a fixed BET)."""

    name: str
    payouts: Mapping[HandRank, int] = field(default_factory=dict)

    def payout_for(self, hand: EvaluatedHand) -> int:
        return self.payouts.get(hand.rank, 0)

    def payout_for_rank(self, rank: HandRank) -> int:
        return self.payouts.get(rank, 0)


@dataclass(frozen=True)
class PlayResult:
    initial_hand: tuple[Card, ...]
    final_hand: tuple[Card, ...]
    evaluated: EvaluatedHand
    payout: int
    profit: int


@dataclass
class VideoPokerGame:
    paytable: Paytable
    evaluate: Callable[[Sequence[Card]], EvaluatedHand]

    def evaluate_hand(self, cards: Sequence[Card]) -> EvaluatedHand:
        return self.evaluate(cards)

    def payout_for_hand(self, cards: Sequence[Card]) -> int:
        return self.paytable.payout_for(self.evaluate_hand(cards))

    def play_hand(
        self,
        hold_mask: int,
        rng: random.Random | None = None,
    ) -> PlayResult:
        rng = rng or random
        deck = make_deck()
        shuffle_deck(deck, rng=rng)
        return self.play_hand_with_state(deck, deal(deck, 5), hold_mask)

    def play_hand_with_state(
        self,
        deck: list[Card],
        hand: Sequence[Card],
        hold_mask: int,
    ) -> PlayResult:
        if len(hand) != 5:
            raise ValueError("expected a 5-card hand")
        initial = tuple(hand)
        final = tuple(apply_hold(initial, hold_mask, deck))
        evaluated = self.evaluate_hand(final)
        payout = self.paytable.payout_for(evaluated)
        return PlayResult(
            initial_hand=initial,
            final_hand=final,
            evaluated=evaluated,
            payout=payout,
            profit=payout - BET,
        )
