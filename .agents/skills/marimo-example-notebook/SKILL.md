---
name: marimo-example-notebook
description: Create or refactor a high-quality marimo example notebook for the wandb/examples repo. Use whenever adding a new example under examples/marimo/, converting a Jupyter example to marimo, or reviewing/refactoring an existing marimo example. Encodes this repo's structure conventions and W&B integration patterns.
---

# marimo example notebooks for wandb/examples

Read [`../marimo-notebook/SKILL.md`](../marimo-notebook/SKILL.md) first for
the marimo file format, reactivity rules, and `marimo check`. If converting
an existing Jupyter notebook, also read
[`../jupyter-to-marimo/SKILL.md`](../jupyter-to-marimo/SKILL.md). This skill
layers repo-specific conventions on top of those.

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

Batch every control into one form so nothing expensive runs until the user
submits:

```python
form = (
    mo.md(
        """
        **Training.**

        {epochs}  {batch_size}
        ...
        """
    )
    .batch(epochs=epochs, batch_size=batch_size, ...)
    .form(submit_button_label="Train model", bordered=False)
)
form
```

`form.value` is `None` until submit. Gate **one** cell on it, with a message
that tells the reader what will happen:

```python
mo.stop(
    form.value is None,
    mo.md("Fill in the form above and click **Train model** to ..."),
)
cfg = form.value
```

Every downstream cell references names defined *after* the gate (`cfg`,
`run`, `model`, ...), so marimo's dependency graph holds them all back until
the form is submitted. Do **not** re-check the button/form in later cells,
wrap cells in `if` guards, or use `mo.ui.run_button()` when a form fits —
one `mo.stop()` replaces all of that.

## Separate logic from presentation

- Heavy lifting (loading data, training, logging, saving artifacts) goes in
  named `@app.function` helpers; the cell body becomes a short, readable
  call: `model, history, final_acc, best_acc = run_training(...)`.
- View cells (`hide_code=True`) render results and contain no logic worth
  reading.
- Push temporaries into functions to keep notebook globals to a minimum —
  marimo notebooks work best with few globals, and every global name is
  reserved across the whole file.
- Present results with real components — `mo.ui.table(rows, selection=None)`
  for tabular results, `mo.callout(..., kind="success"/"warn"/"danger")` for
  status, `mo.vstack` for grouping — not markdown with emoji.

## W&B integration patterns

- **Auth**: offer a `mo.ui.text(kind="password")` API-key field that falls
  back to ambient login (`wandb login`, `WANDB_API_KEY`, netrc) when blank.
  Never write the key into the run config.
- **Re-runs**: marimo keeps the kernel alive across form re-submits, so
  finish any prior run first: `if wandb.run is not None: wandb.finish()`.
- **Entity**: include an entity field and explain that accounts created
  after May 2024 have no personal entity — the run must go to a team.
- **Surface the run URL immediately** after `wandb.init` so readers can
  watch metrics stream: `mo.md(f"**Run started:** [`{run.name}`]({run.url})")`.
- **Expected failures become guidance, not tracebacks.** Wrap only the calls
  that fail for account-setup reasons (`wandb.init`, registry linking) and
  render a `mo.callout(kind="danger")` that names the likely cause and the
  fix. Let everything else fail naturally — no try/except for control flow.
- A recoverable step (e.g. registry linking) should capture its outcome in a
  status value that a separate view cell renders, so the pipeline completes
  either way.
- Group metrics into UI sections with slash-prefixed names
  (`Training/loss`), and put headline numbers in `run.summary`.

## Before handing back

- `uvx marimo check <notebook.py>` passes.
- Globals audit: anything only used inside one step should live in a helper.
- The notebook reads top-to-bottom as a tutorial; every code cell either
  shows output or is a named helper.
- Fresh-eyes test: a reader with a new W&B account can follow Prerequisites,
  submit the form, and verify the result from "Verify and next steps" alone.
