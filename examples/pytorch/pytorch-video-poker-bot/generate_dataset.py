#!/usr/bin/env python3
"""Generate the Jacks or Better training dataset labeled with exact hold EVs."""

from __future__ import annotations

import argparse
from pathlib import Path

from data.generate import DEFAULT_HANDS, DEFAULT_SEED, generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="where to write the dataset (.npz)")
    parser.add_argument("--hands", type=int, default=DEFAULT_HANDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true", help="overwrite existing file")
    args = parser.parse_args()
    generate_dataset(
        args.output,
        hands=args.hands,
        seed=args.seed,
        force=args.force,
    )


if __name__ == "__main__":
    main()
