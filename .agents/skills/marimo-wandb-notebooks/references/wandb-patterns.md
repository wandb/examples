# W&B Patterns

Use these patterns when a marimo example uses the W&B Python SDK.

## Authentication

- Keep exactly one reader-facing Authentication section. Consolidate signup,
  API-key, entity, and login guidance and controls there. Remove duplicate
  standalone Colab-era signup and API-key prompts once their essential guidance
  is represented in this section; preserve unique tutorial-specific context.
- Offer a `mo.ui.text(kind="password")` API-key field.
- Call `wandb.login()` only after the reader explicitly submits the form or
  clicks the run button.
- A fresh molab runtime does not inherit credentials from the reader's local
  computer. When the submitted key is non-empty, pass it to
  `wandb.login(key=...)`. When it is blank, rely on W&B's normal credential
  resolution, including `WANDB_API_KEY` from the marimo Secrets panel, W&B
  settings, or credentials stored by `wandb login` in the current runtime.
- Never display, log, or include the API key in run config. Do not print the
  submitted form value or the full notebook namespace because either can
  expose the key.

## Runs And Reruns

- Prefer a context manager when the run lifecycle fits within one function or
  cell:

  ```python
  with wandb.init() as run:
      run.log({"loss": 0.1})
  ```

### Naming Run Objects in marimo

marimo requires each notebook-global name to have one defining cell. If a
notebook creates W&B runs in multiple cells, do not repeat a top-level
`run = wandb.init(...)` or `with wandb.init() as run:` binding across them.

- Prefer putting each complete run lifecycle in a function or `@app.function`;
  `run` is then an ordinary local name and can be reused naturally.
- When a run remains notebook-global, give each instance a unique, descriptive
  name for its role, such as `training_run`, `evaluation_run`, or
  `artifact_link_run`, and use that name consistently.
- `_run` is a valid cell-local fallback for private plumbing, but do not use
  repeated `_run` bindings as the default in reader-visible teaching code.

When a run cannot use a context manager, explicitly call `.finish()` on the
corresponding run object. If a run must stay active across cells, finish any
prior active run before starting another one.

- Prefer methods on the active run, such as `run.log()`, `run.log_artifact()`,
  and `run.summary`, unless the tutorial intentionally teaches another W&B API
  pattern.
- Within one submission, perform each intended remote write once. On
  re-submission, clean up any prior active run and make stateful remote updates
  idempotent when practical, such as skipping an alias or Registry link that is
  already present.

### Runs Created By Child Processes

When readers need a dashboard link before a blocking command-line program
finishes, preassign the W&B run identity before launching the process:

- Resolve concrete project, entity, and run ID values, then derive the run URL
  from those same values with `wandb.Settings(...).run_url`.
- Start with `os.environ.copy()` and add `WANDB_PROJECT`, `WANDB_ENTITY`, and
  `WANDB_RUN_ID`. Pass that mapping as `env` and use `check=True` with
  `subprocess.run`; never display the environment because it can contain
  secrets.
- Keep preparation, process execution, and URL presentation in separate cells.
  Make the process and URL cells depend on the prepared values, not on each
  other, so the dashboard link can render while the command is still running.
- Do not assume authentication enables a framework's W&B integration. Pass its
  explicit integration option when required—for Hugging Face Trainer commands,
  use `--report_to wandb`—and verify that the expected run is created.
- Pass actual values in the argument list. List-form `subprocess.run` does not
  expand shell variables such as `$WANDB_PROJECT`.

## Ordering Remote Effects

marimo orders cells through name dependencies; it cannot observe mutations to
`wandb.config`, run or Artifact objects, or W&B's remote state.

- If a later cell relies on a W&B write performed by an earlier cell, keep the
  ordered transaction in one cell or helper, or return an immutable result or
  completion value from the writer and make the consumer depend on it. Sharing
  only `wandb`, a run, or an Artifact object does not encode the remote ordering.
- Wait until a write is visible to the server before consuming it. Finish or
  synchronize the run, and wait for an Artifact upload when a later step calls
  `use_artifact`, queries the Public API, or links the Artifact into Registry.
- When reading data that was just logged, capture a stable run or Artifact path
  while it is available, synchronize the producer, and only then issue the
  Public API query.

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

## Media From Remote Filesystems

- Preserve the real media format when adapting a remote file. For example,
  pass MP4 bytes through `io.BytesIO` to `wandb.Video(..., format="mp4")`
  instead of relabeling the source as a GIF.
- Decode audio with a library such as `soundfile` and pass the detected sample
  rate to `wandb.Audio`; do not guess from the tutorial text or source code.
- For OBJ text read from a remote filesystem, use `io.StringIO` and pass
  `file_type="obj"` to `wandb.Object3D`. Some file-like objects expose a
  `.name` that an SDK can mistake for a local path.
- When logging HTML content rather than a local path, pass the text explicitly
  with `data_is_not_path=True`.

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
