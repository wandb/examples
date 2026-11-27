"""Dataset and deterministic scorers shared by evaluation examples."""

import weave

from shared.catalog import filter_games, load_catalog

SCENARIOS = [
    {"players": 2, "minutes": 30, "vibe": "quick and clever"},
    {"players": 4, "minutes": 60, "vibe": "cooperative"},
    {"players": 5, "minutes": 45, "vibe": "social deduction"},
    {"players": 6, "minutes": 90, "vibe": "epic and strategic"},
    # Nothing in the catalog fits this row; the correct answer is "none".
    {"players": 2, "minutes": 10, "vibe": "anything"},
]


@weave.op()
def valid_pick(output: str) -> dict:
    """The reply is exactly one catalog game, or 'none'."""
    names = {game["name"] for game in load_catalog()}
    return {"valid_pick": output == "none" or output in names}


@weave.op()
def fits_constraints(players: int, minutes: int, output: str) -> dict:
    """The pick satisfies the player count and the time budget."""
    fitting = {
        game["name"] for game in filter_games(load_catalog(), players=players, max_minutes=minutes)
    }
    if output == "none":
        return {"fits_constraints": not fitting}
    return {"fits_constraints": output in fitting}
