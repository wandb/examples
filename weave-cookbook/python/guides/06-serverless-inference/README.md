# Serverless Inference

This optional deep dive requires a live model and uses Inference credits. Every model call in the core path goes through [W&B Serverless Inference](https://docs.wandb.ai/inference): an OpenAI-compatible API at `https://api.inference.wandb.ai/v1`, authenticated with the same `WANDB_API_KEY` that logs your traces.

## How it works

[`examples/06_inference.py`](../../examples/06_inference.py) lists the available models, then makes one call with response settings tuned — `temperature=0.2` for stable wording, `max_tokens=60` as a hard cap:

```python
    models = client.models.list()
    print(f"{len(models.data)} models available through one API key; using {model_id()}")

    response = client.chat.completions.create(
        model=model_id(),
        messages=[
            {"role": "system", "content": ANNOUNCER_PROMPT},
            {"role": "user", "content": "Announce that game night starts in ten minutes."},
        ],
        temperature=0.2,
        max_tokens=60,
    )
    print(response.choices[0].message.content)
```

When the provider reports usage, it is available on the response. Switch models by setting `MODEL_ID` in `.env` — every example in this cookbook reads it. Structured output (`response_format` with a JSON schema) is covered in [guide 03](../03-llm-judge/README.md).

## Run it

```bash
uv run python examples/06_inference.py
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

Open the 🍩 link the run prints — the call, with model ID, both messages, and token usage, traced by the OpenAI integration. From that call page, click **Open chat in Playground** to iterate on the same prompt interactively: edit messages, retry, adjust settings, or [compare models side by side](https://docs.wandb.ai/weave/guides/tools/playground). You can also browse and try every model at [wandb.ai/inference](https://wandb.ai/inference).

Full capture: [python-06-inference.txt](../../../assets/expected-output/python-06-inference.txt).
