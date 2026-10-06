# Setup

This cookbook traces a small Game Night Agent with [W&B Weave](https://weave-docs.wandb.ai/). Run everything from `python/`. This first guide logs one conversation — no model call yet, just proof the wiring works.

## How it works

`weave.init` connects the process to the `weave-cookbook` project, creating it on first use. `weave.start_conversation` opens a conversation for `game-night-agent`, and `weave.start_turn` logs one user-agent exchange — the user message in, the recorded reply out. From [`examples/00_hello_trace.py`](../../examples/00_hello_trace.py):

```python
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
```

## Run it

```bash
uv sync
cp .env.example .env   # set WANDB_API_KEY from https://wandb.ai/settings
uv run python examples/00_hello_trace.py
```

Logging to a team? Uncomment `WANDB_ENTITY` in `.env` and set the team name.

## What you should see

```text
weave: Logged in as Weights & Biases user: <your-username>.
weave: View Weave data at https://wandb.ai/<entity>/weave-cookbook/weave
Welcome to game night! Bring snacks.
```

## Inspect it in Weave

Open the link the run prints and switch to **Agents**:

- One agent row, `game-night-agent`, with one conversation.
- Inside it, a single turn: user message `Is game night still on?` and the reply `Welcome to game night! Bring snacks.`

Full capture: [python-00-hello-trace.txt](../../../assets/expected-output/python-00-hello-trace.txt). Problems: [troubleshooting](../../../reference/TROUBLESHOOTING.md).
