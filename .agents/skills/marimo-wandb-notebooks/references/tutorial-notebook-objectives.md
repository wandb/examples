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
- In `marimo run`, render a code snippet for W&B calls that teach the tutorial
  objective when the executable source is otherwise hidden.    

## Narrative Structure

- Keep the notebook readable from top to bottom.
- Preserve useful authorial explanations from the source notebook when they
  still fit the marimo version.
- Start with a clear title and any prerequisites or setup notes the reader
  needs before running the notebook.
- Interleave pipeline code with `## Section` markdown cells that explain what
  the reader is about to run and why it matters.
- Keep code cells purposeful: show a result, teach a core step, or define a
  named helper.
- Move reusable plumbing, model classes, and long utilities into named helpers,
  preferably under a `## Helper functions` section near the bottom.

## Reader Verification

- End with a clear "Verify and next steps" section.
- Tell the reader exactly what to inspect in the W&B UI, including relevant
  charts, tabs, panels, Artifacts, Registry collections, or run summary fields.
- Apply a fresh-eyes test: a reader with a new W&B account should be able to
  follow the setup notes, submit the form, and verify the result from the final
  section alone.
- Suggest one or two natural variations the reader can try next, such as
  changing a hyperparameter, creating a new Artifact version, or comparing runs.
