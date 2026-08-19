# W&B Patterns

Use these patterns when a marimo example uses W&B Python SDK.

## Authentication

- Offer a `mo.ui.text(kind="password")` API-key field.
- When the field is blank, rely on W&B's normal credential resolution
  (`WANDB_API_KEY`, W&B settings, or credentials stored by `wandb login`).
- Never display, log, or store the API key in run config.

## Runs And Reruns

- Prefer a context manager when the run lifecycle fits within one cell:

  ```python
  with wandb.init() as run:
      run.log({"loss": 0.1})
  ```

  Otherwise, explicitly finish the run with `wandb.Run.finish()`.

  If a run must stay active across cells, do not use a context manager. Ensure
  any prior active run is finished before starting another one.

- Prefer run-bound methods such as `wandb.Run.log()`, `wandb.Run.log_artifact()`, and
  `wandb.Run.summary` unless the tutorial intentionally teaches a global
  API from `wandb.apis.public`.

- 

## Entity

- Include an overridable team entity field.
- Explain how to find the appropriate team entity in W&B when needed.

## Expected Failures

- Expected failures should become guidance, not tracebacks.
- Wrap only calls that fail for account-setup reasons, such as `wandb.init()` or
  registry linking.
- Render a `mo.callout(kind="danger")` that names the likely cause and fix.
- Let everything else fail naturally. Do not use `try`/`except` for normal
  control flow.
- A recoverable step, such as registry linking, should capture its outcome in a
  status value that a separate view cell renders, so the pipeline completes
  either way.

## Further reference

For W&B behavior not covered here, prefer the official documentation:

- [W&B documentation](https://docs.wandb.ai)
- [W&B Python SDK reference](https://docs.wandb.ai/models/ref/python)

Use the SDK source only when the documented behavior is insufficient.