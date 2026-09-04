# Tutorial Notebook Objectives

Use this when creating, reviewing, or polishing W&B example notebooks.

## Preserve The Teaching Surface

- Treat these notebooks as tutorials, not just runnable apps.
- Preserve the original lesson flow, explanatory code, and teaching surface
  unless it is broken, duplicative, unsafe, or Jupyter-only.
- Keep customer-facing teaching code close to ordinary Python/Colab style.
  Marimo orchestration should support the lesson without becoming part of the
  code the reader is expected to learn.
- Prefer surgical repairs and incremental marimo cleanup over wholesale
  rewrites.
- Do not collapse the tutorial into opaque helpers. Use named functions when
  they make the taught workflow clearer, but keep the implementation the reader
  is meant to learn inspectable.

## Narrative Structure

- Keep the notebook readable from top to bottom.
- Preserve useful authorial explanations from the source notebook when they
  still fit the marimo version.
- Start with a clear title and any prerequisites or setup notes the reader
  needs before running the notebook.
- Interleave pipeline code with markdown sections that explain what the reader
  is about to run and why it matters.
- Keep code cells purposeful: show a result, teach a core step, or define a
  named helper.
- Move reusable plumbing, model classes, and long utilities into named helpers,
  preferably under a `## Helper functions` section near the bottom.

## Reader Verification

- Preserve the source notebook's ending. Add or adapt concise verification and
  next-step guidance only when readers otherwise lack a clear way to confirm
  the tutorial result; do not append generic boilerplate during a repair or
  exact synchronization.
- When verification guidance is needed, tell the reader exactly what to inspect
  in the W&B UI, including relevant charts, tabs, panels, Artifacts, Registry
  collections, or run summary fields.
- Apply a fresh-eyes test: a reader following the documented prerequisites
  should be able to complete the tutorial and verify the result from the
  notebook's guidance.
