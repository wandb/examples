# W&B Patterns

Use these patterns when a marimo example uses the W&B Python SDK.

## Authentication

- Offer a `mo.ui.text(kind="password")` API-key field.
- When the field is blank, rely on W&B's normal credential resolution
  (`WANDB_API_KEY`, W&B settings, or credentials stored by `wandb login`).
- Never display, log, or include the API key in run config.

## Runs And Reruns

- Prefer a context manager when the run lifecycle fits within one function or
  cell:

  ```python
  with wandb.init() as run:
      run.log({"loss": 0.1})
  ```

Otherwise, explicitly finish the run with `run.finish()`. If a run must stay
active across cells, finish any prior active run before starting another one.

- Prefer methods on the active run, such as `run.log()`, `run.log_artifact()`,
  and `run.summary`, unless the tutorial intentionally teaches another W&B API
  pattern.

## Entity

- Include an overridable team entity field.
- Explain how to find the appropriate team entity in W&B when needed.

## Visible W&B SDK Usage

- Keep W&B SDK usage that teaches the tutorial objective easy for readers to
  inspect.
- Do not bury the featured W&B workflow inside marimo orchestration, callbacks,
  or unrelated plumbing.
- W&B SDK calls may live in a named `@app.function` when a clean reusable
  function better serves the tutorial.
- Move non-teaching plumbing into helpers when it improves the teaching
  surface.

## Expected Failures

- Catch only expected, recoverable W&B failures where the notebook can provide
  actionable guidance, such as authentication, permissions, or Registry setup.
- Render a `mo.callout(kind="danger")` that names the likely cause and fix.
- Let unexpected failures surface; do not use `try`/`except` for normal control
  flow.
- For a recoverable optional step, capture the outcome in a status value and
  render it from a separate view cell.

## Further Reference

For W&B behavior not covered here, prefer the official documentation:

- [W&B documentation](https://docs.wandb.ai/)
- [W&B Python SDK reference](https://docs.wandb.ai/models/ref/python/)

Use SDK source only when the documented behavior is insufficient.
