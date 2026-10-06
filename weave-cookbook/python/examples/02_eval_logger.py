"""Log an evaluation you already ran: no Evaluation object, one prediction at a time."""

import weave

from shared.catalog import filter_games, load_catalog
from shared.config import require_api_key, weave_project

# Outputs captured from an earlier batch run of the agent.
BATCH = [
    {"players": 2, "minutes": 30, "output": "Tidepool"},
    {"players": 4, "minutes": 60, "output": "Gemcutter's Guild"},
    # A bad pick, kept on purpose: Ironroot runs 120 minutes.
    {"players": 3, "minutes": 45, "output": "Ironroot"},
]


def fits_constraints(players: int, minutes: int, output: str) -> bool:
    fitting = {
        game["name"] for game in filter_games(load_catalog(), players=players, max_minutes=minutes)
    }
    return output in fitting


def main() -> None:
    require_api_key()
    weave.init(weave_project())

    logger = weave.EvaluationLogger(
        name="game-night-batch",
        model="game-night-agent",
        dataset="batch-2026-08-02",
    )
    for row in BATCH:
        prediction = logger.log_prediction(
            inputs={"players": row["players"], "minutes": row["minutes"]},
            output=row["output"],
        )
        prediction.log_score(
            scorer="fits_constraints",
            score=fits_constraints(row["players"], row["minutes"], row["output"]),
        )
        prediction.finish()
    logger.log_summary()
    print(f"Logged {len(BATCH)} predictions to evaluation 'game-night-batch'.")


if __name__ == "__main__":
    main()
