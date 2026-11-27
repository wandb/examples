# Serverless Inference

This optional deep dive requires a live model and uses Inference credits. Every model call in the core path goes through [W&B Serverless Inference](https://docs.wandb.ai/inference): an OpenAI-compatible API at `https://api.inference.wandb.ai/v1`, authenticated with the same `WANDB_API_KEY` that logs your traces.

## How it works

[`examples/06-inference.ts`](../../examples/06-inference.ts) lists the available models, then makes one call with response settings tuned — `temperature: 0.2` for stable wording, `max_tokens: 60` as a hard cap:

```typescript
const models = await client.models.list();
console.log(`${models.data.length} models available through one API key; using ${modelId()}`);

const response = await client.chat.completions.create({
  model: modelId(),
  messages: [
    { role: 'system', content: ANNOUNCER_PROMPT },
    { role: 'user', content: 'Announce that game night starts in ten minutes.' },
  ],
  temperature: 0.2,
  max_tokens: 60,
});
console.log(response.choices[0].message.content);
```

When the provider reports usage, it is available on the response. Switch models by setting `MODEL_ID` in `.env` — every example in this cookbook reads it. Structured output (`response_format` with a JSON schema) is covered in [guide 03](../03-llm-judge/README.md).

## Run it

```bash
npm run inference
```

## What you should see

```text
<model count> models available through one API key; using <model ID>
<announcement text>
usage: <prompt tokens> prompt + <completion tokens> completion tokens
```

The model count and wording vary; temperature 0.2 keeps replies stable but not identical.

## Limits worth knowing

- Credits are plan-based, with default spending caps per tier ([usage information and limits](https://docs.wandb.ai/inference/usage-limits)).
- Concurrency is capped per project and per user; exceeding it returns `429 Concurrency limit reached for requests`.

## Inspect it in Weave

Open the printed link and check **Traces** — the `openai.chat.completions.create` call, with model ID, both messages, and token usage, traced by `weave.wrapOpenAI`. From that call page, click **Open chat in Playground** to iterate on the same prompt interactively: edit messages, retry, adjust settings, or [compare models side by side](https://docs.wandb.ai/weave/guides/tools/playground). You can also browse and try every model at [wandb.ai/inference](https://wandb.ai/inference).

Full capture: [typescript-06-inference.txt](../../../assets/expected-output/typescript-06-inference.txt).
