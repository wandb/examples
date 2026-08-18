---
name: marimo-wandb-notebooks
description: Create, convert, review, or refactor repo-ready marimo example notebooks for wandb/examples.
---

# marimo example notebooks for wandb/examples

## Read order

| Situation | Do this |
| --- | --- |
| Always | Read [`references/marimo-idioms.md`](references/marimo-idioms.md) and [`references/tutorial-notebook-objectives.md`](references/tutorial-notebook-objectives.md). |
| Starting from an existing marimo `.py` | Inspect the `.py` and read nearby pipeline logs (`result.json`, `marimo-check.log`, and `marimo-convert.log`) in `.logs/` first. For converted notebooks under `marimo/convert/`, also read [`references/convert-cleanup.md`](references/convert-cleanup.md) before broad cleanup. |
| Starting from `.ipynb` | Run [`../../../scripts/convert-colab-to-marimo.py`](../../../scripts/convert-colab-to-marimo.py) `<notebook.ipynb> --name <example-name>`, then read [`references/convert-cleanup.md`](references/convert-cleanup.md) and the generated `marimo/convert/<name>/.logs/result.json`. |
| Notebook uses W&B | Read [`references/wandb-patterns.md`](references/wandb-patterns.md). |

Preserve tutorial teaching value while applying marimo conventions.

The canonical exemplar is
`examples/marimo/mnist-registry/mnist_registry.py` — when in doubt, match
its structure.

## Existing conversion triage

For notebooks under `marimo/convert/`, diagnose logs before polishing. Read
`.logs/result.json`, inspect the failed stage's log, fix `marimo check`
blockers first, then run a fresh `uvx marimo check`. See
[`references/convert-cleanup.md`](references/convert-cleanup.md) for the full
triage and cleanup checklist.

## Repo conventions

- Each example lives in its own directory: `examples/marimo/<example-name>/`,
  with the notebook as `<example_name>.py`.
- **The `.py` file is the source of truth.** A workflow generates the
  markdown export; never hand-edit a generated `.md` next to a notebook.
- Start the file with a PEP 723 script header (pinned lower bounds, e.g.
  `"marimo>=0.9"`, `"wandb>=0.18"`) followed by a module docstring that says
  what the notebook builds and how to run it:

  ```python
  """One-paragraph summary of what the notebook builds.

  Run:

      uvx marimo edit <example_name>.py --sandbox
  """
  ```

- Runtime droppings (`data/`, `wandb/`, `artifacts/`, `__marimo__/`, model
  weights) must not be committed.

## Notebook structure

Use this marimo skeleton: setup cell, intro, outline, configuration form,
gated pipeline, verify/next steps, helper functions. See
[`references/marimo-idioms.md`](references/marimo-idioms.md) for marimo
mechanics and
[`references/tutorial-notebook-objectives.md`](references/tutorial-notebook-objectives.md)
for narrative structure.

## Core priorities

- Gate expensive work once, then let marimo's graph run.
- Preserve visible teaching code, especially featured W&B API calls.
- Keep notebook globals scarce; move distracting scratch work into helpers.

## Final verification

- `uvx marimo check <notebook.py>` passes.
- Globals audit: anything only used inside one step should live in a helper.
- Tutorial-objectives audit passes: narrative flow, visible API examples, and
  reader verification are preserved.
- `.logs/` files are temporary debugging artifacts and must not be
  referenced by the final notebook or docs.
