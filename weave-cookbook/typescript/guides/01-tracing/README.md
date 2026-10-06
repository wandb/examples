# Tracing

This guide requires a live model and uses Serverless Inference credits. The Game Night Agent searches the local catalog, then asks `meta-llama/Llama-3.1-8B-Instruct` for exactly one game name. Four span calls describe the exchange ([agent tracing docs](https://docs.wandb.ai/weave/guides/tracking/trace-agents)):

| Span call | What it logs | OTel span |
| --- | --- | --- |
| `weave.startConversation` | groups turns under one conversation | (no span) |
| `weave.startTurn` | one user-agent exchange, root of a new trace | `invoke_agent` |
| `weave.startTool` | one tool execution and its result | `execute_tool` |
| `weave.startLLM` | one model call: messages, usage | `chat` |

All of them no-op silently when `weave.init` hasn't run, so instrumented code still works offline.

## How it works

[`examples/01-trace-recommendation.ts`](../../examples/01-trace-recommendation.ts) opens a turn for the user's request, wraps catalog filtering in a tool span (`tool.result` records what the search returned), then runs the model call inside an LLM span. `llm.record({...})` stores the exchange and token usage, and every span is closed in a `finally` block:

```typescript
      const llm = weave.startLLM({ model: modelId(), providerName: 'openai' });
      try {
        const response = await inferenceClient().chat.completions.create({
          model: modelId(),
          messages: [
            { role: 'system', content: SYSTEM_PROMPT },
            { role: 'user', content: prompt },
          ],
        });
        reply = (response.choices[0].message.content ?? '').trim().replace(/^"|"$/g, '');
        llm.record({
          inputMessages: [
            { role: 'system', content: SYSTEM_PROMPT },
            { role: 'user', content: prompt },
          ],
          outputMessages: [{ role: 'assistant', content: reply }],
          usage: {
            inputTokens: response.usage?.prompt_tokens,
            outputTokens: response.usage?.completion_tokens,
          },
        });
      } finally {
        llm.end();
      }
```

`turn.record({...})` then stores the agent's reply on the turn, and `await weave.flushOTel()` sends anything still buffered before the script exits.

## Run it

```bash
npm run recommend -- --players 4 --minutes 60 --vibe "cooperative"
```

## What you should see

```text
The Alchemist's Cellar
```

The candidate list is deterministic; the model's pick varies. This captured pick is useful because the catalog tags *The Alchemist's Cellar* as deduction and puzzle, not cooperative: the trace succeeded, but the recommendation quality did not. Guide 02 measures that difference.

## Inspect it in Weave

Open the printed link and switch to **Agents**:

- `game-night-agent` has a new conversation per run, timestamped in the timeline.
- The turn (`invoke_agent`) carries the user request and the reply, with two child spans:
  - `execute_tool` — `search_catalog` with its JSON arguments and the candidate list.
  - `chat` — the full message list, model ID, and token usage when the provider returns it.

Expected output shape: [typescript-01-trace-recommendation.txt](../../../assets/expected-output/typescript-01-trace-recommendation.txt).

## Useful commands

```bash
npm run recommend -- --players 4 --minutes 10
```

Nothing fits 10 minutes, so the reply is `none` and the turn ends right after the tool span — no `chat` child. Spans mirror control flow.

Add a follow-up to the same conversation:

```bash
npm run recommend -- --players 4 --minutes 60 --vibe "cooperative" --follow-up-minutes 30
```

Expected UI shape: two turns share one conversation ID, while each turn remains its own trace. The application supplies the updated time budget rather than replaying prior model messages. The two-turn control flow is tested offline but has not been re-captured live; its status is recorded in the [source map](../../../reference/SOURCE_MAP.md).
