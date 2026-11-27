"""Load and filter the Game Night board game catalog."""

import json
from pathlib import Path

CATALOG_PATH = Path(__file__).parent / "data" / "games.json"


def load_catalog() -> list[dict]:
    return json.loads(CATALOG_PATH.read_text())


def filter_games(
    games: list[dict],
    *,
    players: int,
    max_minutes: int,
    difficulty: str | None = None,
) -> list[dict]:
    """Return games that fit the player count, time budget, and optional difficulty."""
    return [
        game
        for game in games
        if game["min_players"] <= players <= game["max_players"]
        and game["playtime_minutes"] <= max_minutes
        and (difficulty is None or game["difficulty"] == difficulty)
    ]


def describe_game(game: dict) -> str:
    return (
        f"- {game['name']} ({game['min_players']}-{game['max_players']} players, "
        f"{game['playtime_minutes']} min, {game['difficulty']}): {game['description']}"
    )
