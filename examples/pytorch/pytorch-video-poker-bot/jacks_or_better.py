"""9/6 Jacks or Better: ranking, paytable, and game factory.

`classify` is the single source of truth for hand ranks (vectorized).
`evaluate_hand` is a thin wrapper that returns the richer EvaluatedHand type.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from game import (
    Card,
    EvaluatedHand,
    HandRank,
    Paytable,
    VideoPokerGame,
    hand_to_codes,
)

GAME_ID = "jacks_or_better_9_6"

# Absolute credits paid at BET=5 (royal includes the max-coin bonus).
PAYTABLE = Paytable(
    name="9/6 Jacks or Better",
    payouts={
        HandRank.ROYAL_FLUSH: 4000,
        HandRank.STRAIGHT_FLUSH: 250,
        HandRank.FOUR_OF_A_KIND: 125,
        HandRank.FULL_HOUSE: 45,
        HandRank.FLUSH: 30,
        HandRank.STRAIGHT: 20,
        HandRank.THREE_OF_A_KIND: 15,
        HandRank.TWO_PAIR: 10,
        HandRank.JACKS_OR_BETTER: 5,
        HandRank.NOTHING: 0,
    },
)

# Index into this tuple is what `classify` returns.
RANK_CLASSES: tuple[HandRank, ...] = (
    HandRank.NOTHING,
    HandRank.JACKS_OR_BETTER,
    HandRank.TWO_PAIR,
    HandRank.THREE_OF_A_KIND,
    HandRank.STRAIGHT,
    HandRank.FLUSH,
    HandRank.FULL_HOUSE,
    HandRank.FOUR_OF_A_KIND,
    HandRank.STRAIGHT_FLUSH,
    HandRank.ROYAL_FLUSH,
)
NUM_RANK_CLASSES = len(RANK_CLASSES)

_TEN_RANK = 8
_JACK_RANK = 9
_ACE_RANK = 12


def classify(hands: np.ndarray) -> np.ndarray:
    """Rank many hands at once. Each row is five card codes; result is RANK_CLASSES index."""
    codes = np.asarray(hands)
    if codes.ndim != 2 or codes.shape[1] != 5:
        raise ValueError("expected an (N, 5) array of card codes")
    codes = np.sort(codes, axis=1)

    ranks = codes // 4
    suits = codes % 4
    r0, r1, r2, r3, r4 = (ranks[:, i] for i in range(5))

    is_flush = (suits == suits[:, :1]).all(axis=1)
    distinct = (r0 < r1) & (r1 < r2) & (r2 < r3) & (r3 < r4)
    is_run = distinct & ((r4 - r0) == 4)
    is_wheel = distinct & (r3 == 3) & (r4 == _ACE_RANK)
    is_straight = is_run | is_wheel
    is_royal = is_run & (r0 == _TEN_RANK)

    pair_01, pair_12, pair_23, pair_34 = r0 == r1, r1 == r2, r2 == r3, r3 == r4
    adjacent = (
        pair_01.astype(np.int8)
        + pair_12.astype(np.int8)
        + pair_23.astype(np.int8)
        + pair_34.astype(np.int8)
    )
    has_quads = (r0 == r3) | (r1 == r4)
    has_trips = (r0 == r2) | (r1 == r3) | (r2 == r4)
    is_full_house = (adjacent == 3) & ~has_quads
    is_trips = (adjacent == 2) & has_trips
    is_two_pair = (adjacent == 2) & ~has_trips
    pair_rank = np.where(pair_01, r0, np.where(pair_12, r1, np.where(pair_23, r2, r3)))
    is_high_pair = (adjacent == 1) & (pair_rank >= _JACK_RANK)

    out = np.zeros(codes.shape[0], dtype=np.uint8)
    out[is_high_pair] = RANK_CLASSES.index(HandRank.JACKS_OR_BETTER)
    out[is_two_pair] = RANK_CLASSES.index(HandRank.TWO_PAIR)
    out[is_trips] = RANK_CLASSES.index(HandRank.THREE_OF_A_KIND)
    out[is_straight] = RANK_CLASSES.index(HandRank.STRAIGHT)
    out[is_flush] = RANK_CLASSES.index(HandRank.FLUSH)
    out[is_full_house] = RANK_CLASSES.index(HandRank.FULL_HOUSE)
    out[has_quads] = RANK_CLASSES.index(HandRank.FOUR_OF_A_KIND)
    out[is_flush & is_straight] = RANK_CLASSES.index(HandRank.STRAIGHT_FLUSH)
    out[is_flush & is_royal] = RANK_CLASSES.index(HandRank.ROYAL_FLUSH)
    return out


def evaluate_hand(cards: Sequence[Card]) -> EvaluatedHand:
    """Classify one hand; translate the fast ordinal into EvaluatedHand."""
    if len(cards) != 5:
        raise ValueError("expected exactly 5 cards")
    ordinal = int(classify(np.asarray([hand_to_codes(cards)], dtype=np.uint8))[0])
    return EvaluatedHand(RANK_CLASSES[ordinal])


def make_game() -> VideoPokerGame:
    return VideoPokerGame(paytable=PAYTABLE, evaluate=evaluate_hand)
