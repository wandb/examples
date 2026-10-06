# Log an existing evaluation

This optional recipe is for predictions produced by a batch job or another system. It writes evaluation records to Weave but makes no model call, so it needs `WANDB_API_KEY` without using Inference credits.

## How it works

`logPrediction` and `logScore` are fire-and-forget; `await logSummary()` waits for pending records. [`examples/02-eval-logger.ts`](../../examples/02-eval-logger.ts) logs each prediction and its score:

```typescript
for (const row of BATCH) {
  const prediction = evalLogger.logPrediction(
    { players: row.players, minutes: row.minutes },
    row.output,
  );
  prediction.logScore('fits_constraints', fitsConstraints(row.players, row.minutes, row.output));
  prediction.finish();
}

await evalLogger.logSummary();
```

## Run it

```bash
npm run eval-logger
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

Full capture: [typescript-02-eval-logger.txt](../../../assets/expected-output/typescript-02-eval-logger.txt).
