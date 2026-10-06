"""Evaluate the Game Night Agent: a dataset, two deterministic scorers, one weave.Model."""

import asyncio

import weave

from shared.catalog import describe_game, filter_games, load_catalog
from shared.config import inference_client, model_id, require_api_key, weave_project
from shared.evaluation import SCENARIOS, fits_constraints, valid_pick

SYSTEM_PROMPT = (
    "You are the Game Night Agent. Pick the best game for the group from the candidate list. "
    "Reply with the game name only, exactly as written in the list."
)


class GameNightModel(weave.Model):
    model_name: str
    system_prompt: str

    @weave.op()
    def predict(self, players: int, minutes: int, vibe: str) -> str:
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
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip().strip('"')


def main() -> None:
    require_api_key()
    weave.init(weave_project())

    model = GameNightModel(model_name=model_id(), system_prompt=SYSTEM_PROMPT)
    dataset = weave.Dataset(name="game-night-scenarios", rows=SCENARIOS)
    evaluation = weave.Evaluation(
        dataset=dataset,
        scorers=[valid_pick, fits_constraints],
        evaluation_name="game-night-eval",
    )
    summary = asyncio.run(evaluation.evaluate(model))
    print(summary)


if __name__ == "__main__":
    main()
