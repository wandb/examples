# Log an existing evaluation

This optional recipe is for predictions produced by a batch job or another system. It writes evaluation records to Weave but makes no model call, so it needs `WANDB_API_KEY` without using Inference credits.

## How it works

[`examples/02_eval_logger.py`](../../examples/02_eval_logger.py) logs each existing prediction, attaches its score, and closes the row before publishing the summary:

```python
    for row in BATCH:
        prediction = logger.log_prediction(
            inputs={"players": row["players"], "minutes": row["minutes"]},
            output=row["output"],
        )
        prediction.log_score(
            scorer="fits_constraints",
            score=fits_constraints(row["players"], row["minutes"], row["output"]),
        )
        prediction.finish()
    logger.log_summary()
```

## Run it

```bash
uv run python examples/02_eval_logger.py
```

## What you should see

```text
Logged 3 predictions to evaluation 'game-night-batch'.
```

One batch row deliberately recommends a 120-minute game for a 45-minute request.

## Inspect it in Weave

Open **Evals**, then select `game-night-batch`:

- Three prediction rows appear without rerunning the original application.
- `fits_constraints` is true for two rows and false for the `Ironroot` row.
- The summary records a 2/3 pass rate.

Full capture: [python-02-eval-logger.txt](../../../assets/expected-output/python-02-eval-logger.txt).
