# Prompts

This guide requires live model calls and uses Serverless Inference credits. Prompt wording changes application behavior more than most code changes, so Weave treats prompts as versioned objects ([prompts docs](https://docs.wandb.ai/weave/guides/core-types/prompts)). Two built-in classes support `{placeholder}` parameterization through `format()`:

- `weave.StringPrompt` — one string, for a system message or any standalone text.
- `weave.MessagesPrompt` — a message-list template for chat conversations.

## How it works

[`examples/05_prompts.py`](../../examples/05_prompts.py) defines one of each — a `StringPrompt` with a `{vibe}` placeholder and a `MessagesPrompt` with `{players}`, `{minutes}`, `{candidates}`:

```python
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
```

Both are published (`game-night-system`, `game-night-request`), then the model call is built entirely from `format(...)` output while preserving the name-only application contract:

```python
        messages = [
            {"role": "system", "content": self.system_prompt.format(vibe=vibe)},
            *REQUEST_PROMPT.format(players=players, minutes=minutes, candidates=listing),
        ]
        response = inference_client().chat.completions.create(
            model=self.model_name,
            messages=messages,
        )
        return (response.choices[0].message.content or "").strip().strip('"')
```

At the end the example fetches the published prompt back with `weave.ref("game-night-system").get()` and formats it with a different vibe: application code can load wording by name instead of hardcoding it.

## Run it

```bash
uv run python examples/05_prompts.py
```

## What you should see

```text
The Alchemist's Cellar

fetched game-night-system:latest -> You are the Game Night Agent. Pick exactly one game from the candidate list. Cho...
```

The model's pick varies between runs; the fetched prompt text does not.

## Inspect it in Weave

Open the printed link:

- On the first run, **Prompts** (under Objects) lists `game-night-system` and `game-night-request` at `v0`.
- The fetched `game-night-system` content matches the object used for the model call.
- The 🍩 call link shows the chat completion with fully formatted messages, so you can see exactly what wording reached the model.

Expected output shape: [python-05-prompts.txt](../../../assets/expected-output/python-05-prompts.txt).

## Useful commands

Evaluate two prompt versions against the same named dataset:

```bash
uv run python examples/05_prompts.py --compare
```

```text
baseline: valid_pick=<rate>, fits_constraints=<rate>
vibe-first: valid_pick=<rate>, fits_constraints=<rate>
```

The command publishes both `game-night-system` versions and creates `game-night-prompt-baseline` and `game-night-prompt-vibe-first` in **Evals**. The revised prompt receives the same `{vibe}` value but gives it higher priority. Scores may tie because the deterministic scorers measure validity and constraints; compare row-level selections to see what changed while the dataset and recommendation contract stay fixed.
