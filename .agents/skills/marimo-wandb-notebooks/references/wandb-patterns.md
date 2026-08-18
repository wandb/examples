# W&B Patterns

Use these patterns when a marimo example initializes W&B runs, logs metrics, or
links artifacts and registry entries.

## Authentication

- Offer a `mo.ui.text(kind="password")` API-key field.
- Fall back to ambient login when the field is blank, such as `wandb login`,
  `WANDB_API_KEY`, or netrc.
- Never write the API key into the run config.

## Runs And Reruns

- When possible use context managers when initializing runs.

  ```python
  import wandb

  if wandb.init() as run:
    run.log()
  ```

  If you do not use a context manager, explicitly finish a run with `wandb.Run.finish()`.

- When possible, avoid [global functions](https://docs.wandb.ai/reference), with the exception of `wandb.init()`

- marimo keeps the kernel alive across form re-submits, so finish any prior run
  before starting a new one:

  ```python
  if wandb.run is not None:
      wandb.finish()
  ```

- Surface the run URL immediately after `wandb.init` so readers can watch
  metrics stream:

  ```python
  mo.md(f"**Run started:** [`{run.name}`]({run.url})")
  ```

- Group metrics into UI sections with slash-prefixed names, such as
  `Training/loss`, and put headline numbers in `run.summary`.

## Entity

- Include an entity field.
- Explain that accounts created after May 2024 have no personal entity; the run
  must go to a team.

## Expected Failures

- Expected failures should become guidance, not tracebacks.
- Wrap only calls that fail for account-setup reasons, such as `wandb.init` or
  registry linking.
- Render a `mo.callout(kind="danger")` that names the likely cause and fix.
- Let everything else fail naturally. Do not use `try`/`except` for normal
  control flow.
- A recoverable step, such as registry linking, should capture its outcome in a
  status value that a separate view cell renders, so the pipeline completes
  either way.
