# Evaluations

This guide requires a live model and uses Serverless Inference credits. One relationship drives everything here: **Evaluation = Dataset + Scorers + your application** ([evaluations docs](https://docs.wandb.ai/weave/guides/core-types/evaluations)).

- **Dataset**: versioned rows of scenarios — here, game-night requests.
- **Scorer**: one measurable property of an output. Start deterministic; judges come in the next guide.
- **Evaluation run**: one execution of the dataset against one version of your application.

TypeScript has no class-based `Model` or `Scorer` — you evaluate a function wrapped in `weave.op`. The evaluation calls it with `{ datasetRow }`, and each scorer with `{ datasetRow, modelOutput }`.

Guides 00–01 used agent spans for the **Agents** view. Evaluations use `weave.op` calls in **Evals** and **Traces** so Weave can execute and score the application once per dataset row. The application contract stays the same: one catalog game name or `none`.

## How it works

A scorer is an op that returns one named measurement. The shared dataset and scorers live in [`examples/shared/evaluation.ts`](../../examples/shared/evaluation.ts):

```typescript
const fitsConstraints = weave.op(
  ({ modelOutput, datasetRow }: { modelOutput: string; datasetRow: Scenario }) => {
    const fitting = new Set(
      filterGames(loadCatalog(), {
        players: datasetRow.players,
        maxMinutes: datasetRow.minutes,
      }).map((game) => game.name),
    );
    if (modelOutput === 'none') {
      return { fits_constraints: fitting.size === 0 };
    }
    return { fits_constraints: fitting.has(modelOutput) };
  },
  { name: 'fits_constraints' },
);
```

The agent itself is `gameNightModel`, another `weave.op` function whose OpenAI client is wrapped with `weave.wrapOpenAI`. The `Evaluation` ties all three together:

```typescript
const evaluation = new weave.Evaluation({
  name: 'game-night-eval',
  dataset: new weave.Dataset({ name: 'game-night-scenarios', rows: SCENARIOS }),
  scorers: [validPick, fitsConstraints],
});

const summary = await evaluation.evaluate({ model: gameNightModel });
```

The last dataset row is impossible on purpose; the correct answer is `none`.

## Run it

```bash
npm run evaluate
```

## What you should see

```text
{
  "model_success": {
    "true_count": 5,
    "true_fraction": 1
  },
  "valid_pick": {
    "valid_pick": {
      "true_count": 5,
      "true_fraction": 1
    }
  },
  "fits_constraints": {
    "fits_constraints": {
      "true_count": 5,
      "true_fraction": 1
    }
  },
  "model_latency": {
    "mean": 5.71
  }
}
```

The model can reply with extra words on any run — those rows fail `valid_pick` and the counts dip below 5. Keep them failing; that is what the evaluation is for.

## Inspect it in Weave

Open the 🍩 link the run prints:

- **Evals** lists the run with score columns `valid_pick`, `fits_constraints`, `model_success`, and `model_latency`.
- Click a row to see that scenario's `GameNightModel` call and its `openai.chat.completions.create` child.
- The dataset is a versioned object. Change the system prompt, run again, and compare both runs on the same dataset.

Full capture: [typescript-02-evaluate.txt](../../../assets/expected-output/typescript-02-evaluate.txt).

Already have outputs from another system? Use the optional [EvaluationLogger recipe](evaluation-logger.md) instead of rebuilding that loop here.
