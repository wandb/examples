"""Publish, version, and retrieve objects: the catalog gets an immutable history."""

import weave

from shared.catalog import load_catalog
from shared.config import require_api_key, weave_project

EXPANSION_GAME = {
    "name": "Compost Wars",
    "min_players": 3,
    "max_players": 6,
    "playtime_minutes": 35,
    "difficulty": "light",
    "tags": ["party", "engine-building"],
    "description": "Grow the mightiest compost heap before the first frost.",
}


def main() -> None:
    require_api_key()
    weave.init(weave_project())

    first = weave.publish(load_catalog(), name="game-catalog")
    print(f"published: {first.uri()}")

    expanded = load_catalog() + [EXPANSION_GAME]
    second = weave.publish(expanded, name="game-catalog")
    print(f"published: {second.uri()}")

    latest = weave.ref("game-catalog").get()
    original = weave.ref("game-catalog:v0").get()
    print(f"latest has {len(latest)} games; v0 still has {len(original)}.")


if __name__ == "__main__":
    main()
