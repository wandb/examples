# LLM as a judge

This guide requires live model calls and uses Serverless Inference credits. `fits_constraints` can't tell you whether *Caravan Kings* feels cooperative, so the scorer asks a model one narrow question and returns structured output. A judge is never ground truth; before trusting it, measure how often it agrees with a human.

## How it works

The judge is an ordinary op with three controls: one question, a rubric that excludes everything else, and a strict JSON schema via `response_format` ([structured output docs](https://docs.wandb.ai/inference/response-settings/structured-output)). Successful responses follow the schema; empty content, transport failures, and malformed provider responses still need explicit handling. From [`examples/03_llm_judge.py`](../../examples/03_llm_judge.py):

```python
@weave.op()
def judge_vibe(vibe: str, game: str) -> dict:
    """Narrow LLM judge: one question, structured output."""
    response = inference_client().chat.completions.create(
        model=model_id(),
        messages=[
            {"role": "system", "content": RUBRIC},
            {
                "role": "user",
                "content": f"Requested vibe: {vibe}\nRecommended game: {describe(game)}",
            },
        ],
        response_format=VERDICT_FORMAT,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Judge returned no structured verdict.")
    verdict = json.loads(content)
    if not isinstance(verdict.get("vibe_match"), bool) or not isinstance(
        verdict.get("reason"), str
    ):
        raise ValueError("Judge verdict did not match the expected schema.")
    return verdict
```

The example runs it twice, in order: first an `Evaluation` over six human-labeled rows, where `agrees_with_human` reports the match rate — then as the `vibe_match` scorer in a normal `Evaluation` of `recommend`, carrying the judge's one-sentence reason alongside each score.

## Run it

```bash
uv run python examples/03_llm_judge.py
```

## What you should see

```text
Judge agrees with the human labels on 83% of calibration rows.
{'true_count': 2, 'true_fraction': 0.6666666666666666}
```

Agreement varies between runs — in this captured run the judge missed one human label. With only six rows, this is a calibration signal rather than a confidence interval. If agreement drops, inspect the disagreements and fix the rubric or labels before using the scores.

## Inspect it in Weave

Open the two 🍩 links:

- `vibe-judge-calibration`: six rows; the failing row shows the judge's verdict and `reason` against `human_says`.
- `vibe-judge-eval`: each row's `vibe_match` score with `judge_reason`, and the nested `judge_vibe` call under the scorer — the judge's own LLM call is traced like any other.

Full capture: [python-03-llm-judge.txt](../../../assets/expected-output/python-03-llm-judge.txt).
