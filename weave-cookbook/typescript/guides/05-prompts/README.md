# Prompts

This guide requires live model calls and uses Serverless Inference credits. Prompt wording changes application behavior more than most code changes, so Weave treats prompts as versioned objects ([prompts docs](https://docs.wandb.ai/weave/guides/core-types/prompts)). Two built-in classes support `{placeholder}` parameterization through `format({...})`:

- `weave.StringPrompt` — one string, for a system message or any standalone text.
- `weave.MessagesPrompt` — a message-list template for chat conversations.

## How it works

[`examples/05-prompts.ts`](../../examples/05-prompts.ts) defines one of each — a `StringPrompt` with a `{vibe}` placeholder and a `MessagesPrompt` with `{players}`, `{minutes}`, `{candidates}`:

```typescript
export const systemPrompt = new weave.StringPrompt({
  content:
    'You are the Game Night Agent. Pick exactly one game from the candidate list. ' +
    'Choose the best match for a group that wants {vibe}. ' +
    'Reply with the game name only, exactly as written in the list.',
});

export const requestPrompt = new weave.MessagesPrompt({
  messages: [
    {
      role: 'user',
      content:
        'Players: {players}\nTime available: {minutes} minutes\n\nCandidates:\n{candidates}',
    },
  ],
});
```

Both are published (`game-night-system`, `game-night-request`), then the model call is built entirely from `format({...})` output while preserving the name-only application contract:

```typescript
  const messages = [
    { role: 'system', content: prompt.format({ vibe: scenario.vibe }) },
    ...requestPrompt.format({
      players: scenario.players,
      minutes: scenario.minutes,
      candidates: candidates.map(describeGame).join('\n'),
    }),
  ] as OpenAI.Chat.Completions.ChatCompletionMessageParam[];
  const response = await weave.wrapOpenAI(inferenceClient()).chat.completions.create({
    model: modelId(),
    messages,
  });
  return (response.choices[0].message.content ?? '').trim().replace(/^"|"$/g, '');
```

At the end the example fetches the published prompt back with `StringPrompt.get(client, ref.uri())` and formats it with a different vibe: application code can load wording by reference instead of hardcoding it.

## Run it

```bash
npm run prompts
```

## What you should see

```text
Gemcutter's Guild

fetched game-night-system -> You are the Game Night Agent. Pick exactly one game from the candidate list. Cho...
```

The model's pick varies between runs; the fetched prompt text does not.

## Inspect it in Weave

Open the printed link:

- On the first run, **Prompts** (under Objects) lists `game-night-system` and `game-night-request` at `v0`.
- The fetched `game-night-system` content matches the object used for the model call.
- **Traces** shows the `openai.chat.completions.create` call (the client is wrapped with `weave.wrapOpenAI`) with fully formatted messages, so you can see exactly what wording reached the model.

Expected output shape: [typescript-05-prompts.txt](../../../assets/expected-output/typescript-05-prompts.txt).

## Useful commands

Evaluate two prompt versions against the same named dataset:

```bash
npm run prompts -- --compare
```

```text
baseline: valid_pick=<rate>, fits_constraints=<rate>
vibe-first: valid_pick=<rate>, fits_constraints=<rate>
```

The command publishes both `game-night-system` versions and creates `game-night-prompt-baseline` and `game-night-prompt-vibe-first` in **Evals**. The revised prompt receives the same `{vibe}` value but gives it higher priority. Scores may tie because the deterministic scorers measure validity and constraints; compare row-level selections to see what changed while the dataset and recommendation contract stay fixed.
