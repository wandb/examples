# W&B Patterns

Use these patterns when a marimo example initializes W&B runs, logs metrics, or
links artifacts and registry entries.

## Authentication

- Offer a `mo.ui.text(kind="password")` API-key field.
- Call `wandb.login()` only after the user submits the form.
- Fall back to ambient login when the field is blank, such as `wandb login`,
  `WANDB_API_KEY`, or netrc.
- A molab runtime does not inherit credentials from the reader's computer.
  In molab, use the password field as the interactive login and pass its
  submitted value to `wandb.login(key=...)`. Users can instead set
  `WANDB_API_KEY` in the marimo Secrets panel and leave the field blank.
- Never write the API key into the run config. Do not print the form value,
  widget value, or full notebook namespace because they can contain the key.

## Runs And Reruns

- When possible, initialize runs with context managers.

  ```python
  import wandb

  with wandb.init() as run:
      run.log({"loss": 0.1})
  ```

  If you do not use a context manager, explicitly finish the run with
  `run.finish()`.

- Prefer run-bound methods such as `run.log`, `run.log_artifact`, and
  `run.summary` unless the notebook is intentionally teaching a global API.

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
- Wait for an artifact upload to finish before a later step consumes or links
  the artifact.

## Ordered Writes

- marimo does not track mutations to `wandb.config` or remote run objects.
  Do not spread `wandb.init`, successive config updates, and a Public API
  update across sibling cells that only reference `wandb`.
- Keep an ordered write sequence in one helper or cell, or pass snapshots or
  results between cells so data dependencies encode the order.
- UI controls can update a local preview reactively, but keep each remote
  write behind explicit submission.

## Entity

- Include an entity field.
- Some accounts require a team/entity. Explain how to find the right entity in
  W&B, and make the field easy to override.
- Pass a blank entity or other optional text value as `None`, not as an empty
  string. For example, use `entity=value or None`.

## Results From W&B

When a lesson reads newly logged data through the W&B Public API:

- Capture the run path while the run object is available.
- Query the Public API only after the run finishes and synchronizes.
- Request only the history fields that the presentation uses.
- Keep full precision in W&B. Round only copies prepared for display.
- Guard an empty history before indexing its last row.
- Keep the W&B run page as the primary live-monitoring destination.

For a short introductory workflow, prefer one post-run summary to background
training and repeated polling. Use live refresh only when it supports the W&B
lesson and the producer can yield the kernel.

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
