# Tutorial Notebook Objectives

Use this when creating, reviewing, or polishing W&B example notebooks.

## Preserve The Teaching Surface

- Treat these notebooks as tutorials, not just runnable apps.
- Preserve the original lesson flow, explanatory code, and teaching surface
  unless a specific incompatibility or the user's request requires a change.
- Preserve original comments and docstrings verbatim, including short inline
  comments and comments inside class methods and callbacks. Do not paraphrase,
  shorten, or delete them as a style cleanup. Adapt only comments made inaccurate
  by a required code change; remove obsolete installation or Jupyter directives.
- Audit the whole requested batch against the source `.ipynb` cells, not just
  top-level functions or files already edited. Record any necessary exceptions
  in the work summary rather than adding an unsolicited audit file to the repo.
- Keep customer-facing teaching code close to ordinary Python/Colab style.
  Marimo orchestration should support the lesson without becoming part of the
  code the reader is expected to learn.
- Prefer surgical repairs and incremental marimo cleanup over wholesale
  rewrites.
- Do not collapse the tutorial into opaque helpers. Use named functions when
  they make the taught workflow clearer, but keep the implementation the reader
  is meant to learn inspectable.
- Keep educational configuration beside the step it explains. For example,
  introduce `MODEL_NAME` near model/tokenizer selection and `BLOCK_SIZE` near
  token grouping; uppercase spelling does not make them setup-cell constants.
- Keep model classes, construction, and inspection at their original teaching
  position. A preview such as `model = ConvNet(...)` followed by `model` is part
  of the lesson, even when training later constructs a fresh instance.

## Narrative Structure

- Keep the notebook readable from top to bottom.
- Preserve useful authorial explanations from the source notebook when they
  still fit the marimo version.
- Use exactly one Markdown level-one heading (`#`) for the notebook title. Use
  level-two headings (`##`) for major sections and level-three or deeper
  headings for their subsections without skipping levels. Correct heading
  markers even during an otherwise content-preserving conversion, but do not
  rewrite the heading text or surrounding prose solely to repair the hierarchy.
- Preserve existing heading wording and intentional decorations unless asked
  to restyle them; conversion is not an editorial rewrite.
- Preserve the W&B features banner, branding, explanatory diagram, and useful
  Docs/Resources sections. Reuse the existing notebook pattern, including theme
  support where present; do not replace the banner with a short prose summary.
- Do not put local launch commands such as `uvx marimo edit ... --sandbox`, pip
  instructions, or explanations of script metadata into the reader's molab
  flow. Remove sections that exist only for local installation. Keep relevant
  hardware, download, data, and authentication guidance. Dependency declarations
  belong in PEP 723 metadata; maintainer validation commands belong in this skill.
- Interleave pipeline code with markdown sections that explain what the reader
  is about to run and why it matters.
- Keep code cells purposeful: show a result, teach a core step, or define a
  named helper.
- Move implementation-only plumbing into helpers when useful. Do not move model
  classes or other featured teaching code to a bottom helper section just to
  make the notebook resemble a script.

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
