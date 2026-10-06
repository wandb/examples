# Tracing

This guide requires a live model and uses Serverless Inference credits. The Game Night Agent searches the local catalog, then asks `meta-llama/Llama-3.1-8B-Instruct` for exactly one game name. Four span calls describe the exchange ([agent tracing docs](https://docs.wandb.ai/weave/guides/tracking/trace-agents)):

| Span call | What it logs | OTel span |
| --- | --- | --- |
| `weave.start_conversation` | groups turns under one conversation | (no span) |
| `weave.start_turn` | one user-agent exchange, root of a new trace | `invoke_agent` |
| `weave.start_tool` | one tool execution and its result | `execute_tool` |
| `weave.start_llm` | one model call: messages, usage | `chat` |

All of them no-op silently when `weave.init` hasn't run, so instrumented code still works offline.

## How it works

[`examples/01_trace_recommendation.py`](../../examples/01_trace_recommendation.py) opens a turn for the user's request and wraps catalog filtering in a tool span, recording the result on it:

```python
    with weave.start_turn(user_message=request) as turn:
        with weave.start_tool(
            name="search_catalog",
            arguments=json.dumps({"players": players, "max_minutes": minutes}),
        ) as tool:
            candidates = filter_games(load_catalog(), players=players, max_minutes=minutes)
            tool.result = json.dumps([game["name"] for game in candidates])
```

When candidates exist, the model call runs inside an LLM span, and `llm.record(...)` stores the exchange and token usage:

```python
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
```

`turn.record(...)` then stores the agent's reply on the turn. `weave.init` runs with `implicitly_patch_integrations: False` so the model call is logged once by the LLM span, not a second time by the OpenAI integration.

## Run it

```bash
uv run python examples/01_trace_recommendation.py --players 4 --minutes 60 --vibe "cooperative"
```

## What you should see

```text
Gemcutter's Guild
```

The candidate list is deterministic; the model's pick varies. This captured pick is useful because the catalog tags *Gemcutter's Guild* as strategy and engine-building, not cooperative: the trace succeeded, but the recommendation quality did not. Guide 02 measures that difference.

## Inspect it in Weave

Open the printed link and switch to **Agents**:

- `game-night-agent` has a new conversation per run, timestamped in the timeline.
- The turn (`invoke_agent`) carries the user request and the reply, with two child spans:
  - `execute_tool` — `search_catalog` with its JSON arguments and the candidate list.
  - `chat` — the full message list, model ID, and token usage when the provider returns it.

Expected output shape: [python-01-trace-recommendation.txt](../../../assets/expected-output/python-01-trace-recommendation.txt).

## Useful commands

```bash
uv run python examples/01_trace_recommendation.py --players 4 --minutes 10
```

Nothing fits 10 minutes, so the reply is `none` and the turn ends right after the tool span — no `chat` child. Spans mirror control flow.

Add a follow-up to the same conversation:

```bash
uv run python examples/01_trace_recommendation.py --players 4 --minutes 60 --vibe "cooperative" --follow-up-minutes 30
```

Expected UI shape: two turns share one conversation ID, while each turn remains its own trace. The application supplies the updated time budget rather than replaying prior model messages. The two-turn control flow is tested offline but has not been re-captured live; its status is recorded in the [source map](../../../reference/SOURCE_MAP.md).
