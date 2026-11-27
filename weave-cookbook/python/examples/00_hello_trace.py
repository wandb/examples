"""Log one conversation to Weave. No model call, no catalog -- just proof the wiring works."""

import os

import weave
from dotenv import load_dotenv
from weave.conversation import Message

load_dotenv()


def main() -> None:
    if not os.getenv("WANDB_API_KEY"):
        raise SystemExit(
            "WANDB_API_KEY is not set. Copy .env.example to .env and add the key "
            "from https://wandb.ai/settings."
        )
    project = os.getenv("WANDB_PROJECT", "weave-cookbook")
    entity = os.getenv("WANDB_ENTITY")
    weave.init(f"{entity}/{project}" if entity else project)

    with weave.start_conversation(agent_name="game-night-agent"):
        with weave.start_turn(user_message="Is game night still on?") as turn:
            reply = "Welcome to game night! Bring snacks."
            turn.record(output_messages=[Message(role="assistant", content=reply)])
            print(reply)


if __name__ == "__main__":
    main()
