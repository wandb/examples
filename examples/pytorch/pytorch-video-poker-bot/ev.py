"""Exact expected-value calculator for hold decisions.

Uses Jacks or Better's vectorized classifier to precompute subset tables,
then answers EV for every hold mask with a handful of lookups.
"""

from __future__ import annotations

from itertools import chain, combinations
from math import comb
from typing import Sequence

import numpy as np

from game import BET, CARDS_PER_HAND, DECK_SIZE, NUM_HOLDS, Card, Paytable, hand_to_codes
from jacks_or_better import NUM_RANK_CLASSES, RANK_CLASSES, classify

TOTAL_HANDS = comb(DECK_SIZE, CARDS_PER_HAND)
ALL_POSITIONS = NUM_HOLDS - 1  # every card position set
CARDS_LEFT_IN_DECK = DECK_SIZE - CARDS_PER_HAND

_BINOMIAL = np.array(
    [[comb(n, k) for k in range(CARDS_PER_HAND + 1)] for n in range(DECK_SIZE + 1)],
    dtype=np.int64,
)


def colex_index(sorted_codes: np.ndarray, size: int) -> np.ndarray:
    index = np.zeros(sorted_codes.shape[0], dtype=np.int64)
    for position in range(size):
        index += _BINOMIAL[sorted_codes[:, position], position + 1]
    return index


def colex_index_scalar(sorted_codes: Sequence[int]) -> int:
    return sum(comb(code, position + 1) for position, code in enumerate(sorted_codes))


class EVTables:
    def __init__(self, subset_counts: Sequence[np.ndarray], hand_ranks: np.ndarray) -> None:
        self._subset_counts = tuple(subset_counts)
        self._hand_ranks = hand_ranks

    def counts_for(self, sorted_codes: Sequence[int]) -> np.ndarray:
        size = len(sorted_codes)
        index = colex_index_scalar(sorted_codes)
        if size == CARDS_PER_HAND:
            counts = np.zeros(NUM_RANK_CLASSES, dtype=np.int64)
            counts[int(self._hand_ranks[index])] = 1
            return counts
        return self._subset_counts[size][index].astype(np.int64)


def build_tables() -> EVTables:
    flat = np.fromiter(
        chain.from_iterable(combinations(range(DECK_SIZE), CARDS_PER_HAND)),
        dtype=np.int8,
        count=TOTAL_HANDS * CARDS_PER_HAND,
    )
    hands = flat.reshape(TOTAL_HANDS, CARDS_PER_HAND)
    ranks = classify(hands).astype(np.int64)

    hand_ranks = np.zeros(TOTAL_HANDS, dtype=np.uint8)
    hand_ranks[colex_index(hands, CARDS_PER_HAND)] = ranks.astype(np.uint8)

    subset_counts: list[np.ndarray] = []
    for size in range(CARDS_PER_HAND):
        rows = comb(DECK_SIZE, size)
        table = np.zeros(rows * NUM_RANK_CLASSES, dtype=np.int64)
        for positions in combinations(range(CARDS_PER_HAND), size):
            if size == 0:
                index = np.zeros(TOTAL_HANDS, dtype=np.int64)
            else:
                index = colex_index(hands[:, positions], size)
            table += np.bincount(
                index * NUM_RANK_CLASSES + ranks,
                minlength=rows * NUM_RANK_CLASSES,
            )
        subset_counts.append(table.reshape(rows, NUM_RANK_CLASSES).astype(np.uint32))
    return EVTables(subset_counts, hand_ranks)


_TABLES: EVTables | None = None


def get_tables() -> EVTables:
    """Build EV tables once per process; reuse in memory afterward."""
    global _TABLES
    if _TABLES is None:
        _TABLES = build_tables()
    return _TABLES


def rank_payouts(paytable: Paytable) -> np.ndarray:
    """Reward for each hand rank, in the order `classify` numbers them."""
    return np.array([paytable.payout_for(rank) for rank in RANK_CLASSES], dtype=np.int64)


def hold_expected_values(hand: Sequence[Card], payouts: np.ndarray) -> list[float]:
    """Exact expected profit (in credits) for every hold mask.

    `payouts` comes from `rank_payouts`; build it once and reuse it.
    """
    tables = get_tables()
    codes = hand_to_codes(hand)

    subset_payout = [0] * NUM_HOLDS
    for subset in range(NUM_HOLDS):
        sorted_codes = sorted(
            codes[position] for position in range(CARDS_PER_HAND) if subset & (1 << position)
        )
        subset_payout[subset] = int(tables.counts_for(sorted_codes) @ payouts)

    expected_values: list[float] = []
    for hold_mask in range(NUM_HOLDS):
        discards = ALL_POSITIONS ^ hold_mask
        total = 0
        subset = discards
        while True:
            sign = -1 if bin(subset).count("1") % 2 else 1
            total += sign * subset_payout[hold_mask | subset]
            if subset == 0:
                break
            subset = (subset - 1) & discards
        draws = comb(CARDS_LEFT_IN_DECK, bin(discards).count("1"))
        expected_values.append(total / draws - BET)
    return expected_values
