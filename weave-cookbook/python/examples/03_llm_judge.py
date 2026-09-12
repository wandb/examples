"""An LLM judge for the one property code can't measure: does the pick match the vibe?"""

import asyncio
import json

import weave

from shared.catalog import describe_game, filter_games, load_catalog
from shared.config import inference_client, model_id, require_api_key, weave_project

RUBRIC = (
    "You judge board game recommendations. Decide only whether the game matches the "
    "requested vibe. Judge mood and play style; ignore player count and duration. "
    "Return vibe_match and a one-sentence reason."
)

VERDICT_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "VibeVerdict",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "vibe_match": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["vibe_match", "reason"],
            "additionalProperties": False,
        },
    },
}

# Human-labeled rows. The judge has to agree with these before its scores mean anything.
CALIBRATION = [
    {"vibe": "cooperative", "game": "Moonbase Delta", "human_says": True},
    {"vibe": "cooperative", "game": "Caravan Kings", "human_says": False},
    {"vibe": "quick party fun", "game": "Orchard Sprint", "human_says": True},
    {"vibe": "quick party fun", "game": "Ironroot", "human_says": False},
    {"vibe": "brainy deduction", "game": "The Alchemist's Cellar", "human_says": True},
    {"vibe": "brainy deduction", "game": "Sky Ferry", "human_says": False},
]

SCENARIOS = [
    {"players": 4, "minutes": 60, "vibe": "cooperative"},
    {"players": 2, "minutes": 45, "vibe": "brainy deduction"},
    {"players": 5, "minutes": 30, "vibe": "quick party fun"},
]

SYSTEM_PROMPT = (
    "You are the Game Night Agent. Pick the best game for the group from the candidate list. "
    "Reply with the game name only, exactly as written in the list."
)


def describe(game_name: str) -> str:
    for game in load_catalog():
        if game["name"] == game_name:
            return describe_game(game)
    return game_name


@weave.op()
def judge_vibe(vibe: str, game: str) -> dict:
    """Narrow LLM judge: one question, structured output."""
    response = inference_client().chat.completions.create(
        model=model_id(),
        messages=[
            {"role": "system", "content": RUBRIC},
            {
                "role": "user",
                "content": f"Requested vibe: {vibe}\nRecommended game: {describe(game)}",
            },
        ],
        response_format=VERDICT_FORMAT,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Judge returned no structured verdict.")
    verdict = json.loads(content)
    if not isinstance(verdict.get("vibe_match"), bool) or not isinstance(
        verdict.get("reason"), str
    ):
        raise ValueError("Judge verdict did not match the expected schema.")
    return verdict


@weave.op()
def agrees_with_human(human_says: bool, output: dict) -> dict:
    return {"agrees_with_human": output["vibe_match"] == human_says}


@weave.op()
def vibe_match(vibe: str, output: str) -> dict:
    """Scorer that delegates to the judge. Not ground truth -- a calibrated opinion."""
    verdict = judge_vibe(vibe, output)
    return {"vibe_match": verdict["vibe_match"], "judge_reason": verdict["reason"]}


@weave.op()
def recommend(players: int, minutes: int, vibe: str) -> str:
    candidates = filter_games(load_catalog(), players=players, max_minutes=minutes)
    if not candidates:
        return "none"
    prompt = (
        f"Players: {players}\n"
        f"Time available: {minutes} minutes\n"
        f"Vibe: {vibe}\n\n"
        "Candidates:\n" + "\n".join(describe_game(game) for game in candidates)
    )
    response = inference_client().chat.completions.create(
        model=model_id(),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip().strip('"')


def main() -> None:
    require_api_key()
    weave.init(weave_project())

    calibration = weave.Evaluation(
        dataset=CALIBRATION,
        scorers=[agrees_with_human],
        evaluation_name="vibe-judge-calibration",
    )
    summary = asyncio.run(calibration.evaluate(judge_vibe))
    agreement = summary["agrees_with_human"]["agrees_with_human"]["true_fraction"]
    print(f"Judge agrees with the human labels on {agreement:.0%} of calibration rows.")

    agent_eval = weave.Evaluation(
        dataset=SCENARIOS,
        scorers=[vibe_match],
        evaluation_name="vibe-judge-eval",
    )
    agent_summary = asyncio.run(agent_eval.evaluate(recommend))
    print(agent_summary["vibe_match"]["vibe_match"])


if __name__ == "__main__":
    main()
