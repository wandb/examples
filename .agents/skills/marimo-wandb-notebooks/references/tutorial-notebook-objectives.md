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

## Visible API Examples

- Keep short, pedagogically important W&B calls visible in narrative code cells
  when the notebook is teaching those calls step by step. Common examples
  include `wandb.init`, `wandb.Artifact`, `run.use_artifact()`,
  `run.log_artifact()`, `run.log()`, and `run.summary`.
- Move reusable or distracting implementation detail, such as long data-loading
  utilities, model classes, repeated training loops, and path handling, into
  named helpers. Keep this detail if it serves to teach how to use W&B.
- Fix marimo graph issues while preserving the reader's ability to inspect the
  featured library calls.

## Reader Verification

- End with a clear "Verify and next steps" section.
- Tell the reader exactly what to inspect in the W&B UI, including relevant
  charts, tabs, panels, Artifacts, Registry collections, or run summary fields.
- Apply a fresh-eyes test: a reader with a new W&B account should be able to
  follow the setup notes, submit the form, and verify the result from the final
  section alone.
- Suggest one or two natural variations the reader can try next, such as
  changing a hyperparameter, creating a new Artifact version, or comparing runs.
