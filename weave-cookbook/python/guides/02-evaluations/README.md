# Evaluations

This guide requires a live model and uses Serverless Inference credits. One relationship drives everything here: **Evaluation = Dataset + Scorers + your application** ([evaluations docs](https://docs.wandb.ai/weave/guides/core-types/evaluations)).

- **Dataset**: versioned rows of scenarios — here, game-night requests.
- **Scorer**: one measurable property of an output. Start deterministic; judges come in the next guide.
- **Evaluation run**: one execution of the dataset against one version of your application.

Guides 00–01 used agent spans for the **Agents** view. Evaluations use `weave.op` calls in **Evals** and **Traces** so Weave can execute and score the application once per dataset row. The application contract stays the same: one catalog game name or `none`.

## How it works

A scorer is an op that returns one named measurement — inputs come from the dataset row, `output` from the application. The shared dataset and scorers live in [`examples/shared/evaluation.py`](../../examples/shared/evaluation.py):

```python
@weave.op()
def fits_constraints(players: int, minutes: int, output: str) -> dict:
    """The pick satisfies the player count and the time budget."""
    fitting = {
        game["name"] for game in filter_games(load_catalog(), players=players, max_minutes=minutes)
    }
    if output == "none":
        return {"fits_constraints": not fitting}
    return {"fits_constraints": output in fitting}
```

The agent is wrapped in a `weave.Model` subclass — its attributes (`model_name`, `system_prompt`) are captured and versioned, and `predict` is the tracked entry point. The `Evaluation` ties all three together:

```python
    model = GameNightModel(model_name=model_id(), system_prompt=SYSTEM_PROMPT)
    dataset = weave.Dataset(name="game-night-scenarios", rows=SCENARIOS)
    evaluation = weave.Evaluation(
        dataset=dataset,
        scorers=[valid_pick, fits_constraints],
        evaluation_name="game-night-eval",
    )
    summary = asyncio.run(evaluation.evaluate(model))
```

The last dataset row is impossible on purpose; the correct answer is `none`.

## Run it

```bash
uv run python examples/02_evaluate.py
```

## What you should see

```text
{'valid_pick': {'valid_pick': {'true_count': 5, 'true_fraction': 1.0}}, 'fits_constraints': {'fits_constraints': {'true_count': 5, 'true_fraction': 1.0}}, 'model_latency': {'mean': 0.71}}
```

The model can reply with extra words on any run — those rows fail `valid_pick` and the counts dip below 5. Keep them failing; that is what the evaluation is for.

## Inspect it in Weave

Open the 🍩 link the run prints:

- **Evals** lists the run with score columns `valid_pick`, `fits_constraints`, and `model_latency`.
- Click a row to see that scenario's `predict` call and its `openai.chat.completions.create` child.
- `GameNightModel` and `game-night-scenarios` are versioned objects. Change `system_prompt`, run again, and compare both runs on the same named dataset.

Full capture: [python-02-evaluate.txt](../../../assets/expected-output/python-02-evaluate.txt).

Already have outputs from another system? Use the optional [EvaluationLogger recipe](evaluation-logger.md) instead of rebuilding that loop here.
