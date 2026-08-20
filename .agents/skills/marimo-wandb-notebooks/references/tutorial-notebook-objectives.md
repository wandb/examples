# Tutorial Notebook Objectives

Use this when creating, reviewing, or polishing W&B example notebooks.

## Preserve The Teaching Surface

- Treat these notebooks as tutorials, not just runnable apps.
- Preserve the original lesson flow, explanatory code, and visible W&B API
  usage unless it is broken, duplicative, unsafe, or Jupyter-only.
- Prefer surgical repairs and incremental marimo cleanup over wholesale
  rewrites.
- Do not collapse a tutorial into one opaque pipeline helper. A reader should
  be able to see how the featured W&B workflow is implemented.

## Narrative Structure

- Keep the notebook readable from top to bottom.
- Preserve useful authorial explanations from the source notebook when they
  still fit the marimo version.
- Start with a clear title and any prerequisites or setup notes the reader
  needs before running the notebook.
- Interleave pipeline code with `## Section` markdown cells that explain what
  the reader is about to run and why it matters.
- Make each code cell justify its place in the tutorial: it should show output,
  teach a core API step, or define a named helper.
- Put reusable plumbing, model classes, and long utilities under a
  `## Helper functions` section near the bottom.

## Visible W&B SDK Calls

- Keep all W&B Python SDK calls in visible code cells. This includes calls
  through `wandb` and SDK objects such as runs, Artifacts, tables, Registry
  objects, and API clients.
- Do not hide W&B SDK calls in helpers, callbacks, or other abstractions.
  Readers should be able to inspect the SDK code that performs each W&B
  operation.
- Gate side-effecting steps without hiding them. Put buttons/forms in small
  control cells and use `mo.stop(...)` in the visible implementation cell
  before the guarded W&B calls.
- Verify the rendered app view with `marimo run --include-code` when the
  notebook is intended to teach W&B APIs. Without `--include-code`, `marimo run`
  hides source code by default and can make a correct notebook look like a
  button-only app.
- If the deployment target cannot expose code cells, add explicit markdown code
  snippets for the W&B SDK calls so readers can still inspect the API usage in
  the rendered tutorial.
- Move only non-W&B implementation detail into helpers when useful for
  readability, such as data loading, model definitions, repeated training
  logic, or path handling.

## Reader Verification

- End with a clear "Verify and next steps" section.
- Tell the reader exactly what to inspect in the W&B UI, including relevant
  charts, tabs, panels, Artifacts, Registry collections, or run summary fields.
- Apply a fresh-eyes test: a reader with a new W&B account should be able to
  follow the setup notes, submit the form, and verify the result from the final
  section alone.
- Suggest one or two natural variations the reader can try next, such as
  changing a hyperparameter, creating a new Artifact version, or comparing runs.
