---
name: marimo-wandb-notebooks
description: Create, convert, review, or refactor repo-ready marimo example notebooks for wandb/examples.
---

# marimo example notebooks for wandb/examples

## Read order

| Situation | Do this |
| --- | --- |
| Always | Read [`references/marimo-idioms.md`](references/marimo-idioms.md). |
| Starting from an existing marimo `.py` | Do not run `prepare-marimo-example.py`. Inspect the `.py`, run `uvx marimo check <notebook.py>`, and polish against the repo conventions below. |
| Starting from `.ipynb` | Run [`../../../scripts/prepare-marimo-example.py`](../../../scripts/prepare-marimo-example.py) `<notebook.ipynb> --name <example-name>`, then read [`references/conversion-cleanup.md`](references/conversion-cleanup.md) with `.conversion/conversion-report.md`, `.conversion/marimo-convert.txt`, `.conversion/marimo-check.txt`, and `.conversion/conversion.json`. |
| Notebook uses W&B | Read [`references/wandb-patterns.md`](references/wandb-patterns.md). |

The canonical exemplar is
`examples/marimo/mnist-registry/mnist_registry.py` — when in doubt, match
its structure.

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

Order the notebook as a narrative the reader scrolls through top to bottom:

1. **Setup cell** — `with app.setup(hide_code=True):` holds all imports,
   constants, and environment detection (e.g. device selection). One place,
   not scattered across cells.
2. **Intro prose** — a markdown cell (`hide_code=True`) with the title,
   "What you will build", and "Prerequisites". Prose goes in markdown cells,
   never mixed into code cells.
3. **`mo.outline()`** in its own cell near the top, so readers see the
   notebook's shape at a glance.
4. **Configuration** — all UI controls batched into a single submittable
   form (see below).
5. **The pipeline** — logic cells that consume the form, interleaved with
   `## Section` markdown cells.
6. **Verify and next steps** — a closing markdown cell telling the reader
   exactly what to look at (in the W&B UI: which charts, tabs, panels) and
   what to try next.
7. **Helper functions** — `@app.function` / `@app.class_definition` cells
   under a `## Helper functions` section at the bottom.

## Gate execution once, then let the graph run

Batch expensive controls into one form, gate once with `mo.stop`, and let
downstream cells depend on names defined after the gate. See
[`references/marimo-idioms.md`](references/marimo-idioms.md) for the detailed
pattern.

## Separate logic from presentation

Put heavy work in named helpers and keep view cells focused on rendering. See
[`references/marimo-idioms.md`](references/marimo-idioms.md) for details.

## Final verification

- `uvx marimo check <notebook.py>` passes.
- Globals audit: anything only used inside one step should live in a helper.
- The notebook reads top-to-bottom as a tutorial; every code cell either
  shows output or is a named helper.
- Fresh-eyes test: a reader with a new W&B account can follow Prerequisites,
  submit the form, and verify the result from "Verify and next steps" alone.
- `.conversion/` files are temporary debugging artifacts and must not be
  referenced by the final notebook or docs.
