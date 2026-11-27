"""Game Night Agent: one conversation with traced tool and LLM work per turn."""

import argparse
import json

import weave
from weave.conversation import Message, Usage

from shared.catalog import describe_game, filter_games, load_catalog
from shared.config import inference_client, model_id, require_api_key, weave_project

SYSTEM_PROMPT = (
    "You are the Game Night Agent. Pick exactly one game from the candidate list. "
    "Reply with the game name only, exactly as written in the list. "
    "Never invent a game that is not on the list."
)


def recommend_game(
    players: int,
    minutes: int,
    vibe: str,
    user_message: str | None = None,
) -> str:
    request = user_message or (
        f"Recommend a game for {players} players, {minutes} minutes, vibe: {vibe}."
    )
    with weave.start_turn(user_message=request) as turn:
        with weave.start_tool(
            name="search_catalog",
            arguments=json.dumps({"players": players, "max_minutes": minutes}),
        ) as tool:
            candidates = filter_games(load_catalog(), players=players, max_minutes=minutes)
            tool.result = json.dumps([game["name"] for game in candidates])

        if not candidates:
            reply = "none"
        else:
            prompt = (
                f"Players: {players}\n"
                f"Time available: {minutes} minutes\n"
                f"Vibe: {vibe}\n\n"
                "Candidates:\n" + "\n".join(describe_game(game) for game in candidates)
            )
            with weave.start_llm(model=model_id(), provider_name="openai") as llm:
                response = inference_client().chat.completions.create(
                    model=model_id(),
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                reply = (response.choices[0].message.content or "").strip().strip('"')
                usage = response.usage
                llm.record(
                    input_messages=[
                        Message(role="system", content=SYSTEM_PROMPT),
                        Message(role="user", content=prompt),
                    ],
                    output_messages=[Message(role="assistant", content=reply)],
                    usage=(
                        Usage(
                            input_tokens=usage.prompt_tokens,
                            output_tokens=usage.completion_tokens,
                        )
                        if usage is not None
                        else None
                    ),
                )

        turn.record(output_messages=[Message(role="assistant", content=reply)])
        return reply


def run_conversation(
    players: int,
    minutes: int,
    vibe: str,
    follow_up_minutes: int | None = None,
) -> list[str]:
    replies = []
    with weave.start_conversation(agent_name="game-night-agent", model=model_id()):
        replies.append(recommend_game(players, minutes, vibe))
        if follow_up_minutes is not None:
            replies.append(
                recommend_game(
                    players,
                    follow_up_minutes,
                    vibe,
                    user_message=f"What if we only have {follow_up_minutes} minutes?",
                )
            )
    return replies


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend a board game for game night.")
    parser.add_argument("--players", type=int, default=4, help="number of players")
    parser.add_argument("--minutes", type=int, default=60, help="time budget in minutes")
    parser.add_argument("--vibe", default="friendly and fun", help="mood the group is after")
    parser.add_argument(
        "--follow-up-minutes",
        type=int,
        help="log a second turn in the same conversation with a new time budget",
    )
    args = parser.parse_args()

    require_api_key()
    # The LLM span below records the model call; op-level autopatching would log it twice.
    weave.init(weave_project(), settings={"implicitly_patch_integrations": False})

    for reply in run_conversation(
        args.players,
        args.minutes,
        args.vibe,
        follow_up_minutes=args.follow_up_minutes,
    ):
        print(reply)


if __name__ == "__main__":
    main()
