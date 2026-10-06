"""Parameterized, versioned prompts: the agent's wording becomes a tracked object."""

import argparse
import asyncio

import weave

from shared.catalog import describe_game, filter_games, load_catalog
from shared.config import inference_client, model_id, require_api_key, weave_project
from shared.evaluation import SCENARIOS, fits_constraints, valid_pick

SYSTEM_PROMPT = weave.StringPrompt(
    "You are the Game Night Agent. Pick exactly one game from the candidate list. "
    "Choose the best match for a group that wants {vibe}. "
    "Reply with the game name only, exactly as written in the list."
)

REQUEST_PROMPT = weave.MessagesPrompt(
    [
        {
            "role": "user",
            "content": (
                "Players: {players}\nTime available: {minutes} minutes\n\nCandidates:\n{candidates}"
            ),
        }
    ]
)

VIBE_FIRST_SYSTEM_PROMPT = weave.StringPrompt(
    "You are the Game Night Agent. Pick exactly one game from the candidate list. "
    "Prioritize the requested vibe ({vibe}), then difficulty. "
    "Reply with the game name only, exactly as written in the list."
)


class PromptedGameNightModel(weave.Model):
    model_name: str
    system_prompt: str

    @weave.op()
    def predict(self, players: int, minutes: int, vibe: str) -> str:
        candidates = filter_games(load_catalog(), players=players, max_minutes=minutes)
        if not candidates:
            return "none"
        listing = "\n".join(describe_game(game) for game in candidates)
        messages = [
            {"role": "system", "content": self.system_prompt.format(vibe=vibe)},
            *REQUEST_PROMPT.format(players=players, minutes=minutes, candidates=listing),
        ]
        response = inference_client().chat.completions.create(
            model=self.model_name,
            messages=messages,
        )
        return (response.choices[0].message.content or "").strip().strip('"')


def compare_prompts() -> None:
    weave.publish(VIBE_FIRST_SYSTEM_PROMPT, name="game-night-system")
    dataset = weave.Dataset(name="game-night-scenarios", rows=SCENARIOS)
    prompts = [
        ("baseline", SYSTEM_PROMPT),
        ("vibe-first", VIBE_FIRST_SYSTEM_PROMPT),
    ]
    for label, prompt in prompts:
        evaluation = weave.Evaluation(
            dataset=dataset,
            scorers=[valid_pick, fits_constraints],
            evaluation_name=f"game-night-prompt-{label}",
        )
        model = PromptedGameNightModel(model_name=model_id(), system_prompt=prompt.content)
        summary = asyncio.run(evaluation.evaluate(model))
        valid = summary["valid_pick"]["valid_pick"]["true_fraction"]
        fits = summary["fits_constraints"]["fits_constraints"]["true_fraction"]
        print(f"{label}: valid_pick={valid:.0%}, fits_constraints={fits:.0%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish and compare versioned prompts.")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="evaluate two prompt versions against game-night-scenarios",
    )
    args = parser.parse_args()

    require_api_key()
    weave.init(weave_project())

    weave.publish(SYSTEM_PROMPT, name="game-night-system")
    weave.publish(REQUEST_PROMPT, name="game-night-request")

    if args.compare:
        compare_prompts()
        return

    players, minutes, vibe = 4, 60, "cooperative"
    model = PromptedGameNightModel(model_name=model_id(), system_prompt=SYSTEM_PROMPT.content)
    print(model.predict(players, minutes, vibe))

    # Application code can load wording by name instead of hardcoding it.
    fetched = weave.ref("game-night-system").get()
    print(f"\nfetched game-night-system:latest -> {fetched.format(vibe='competitive')[:80]}...")


if __name__ == "__main__":
    main()
