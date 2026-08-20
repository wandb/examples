# Marimo Idioms

Use this as a review checklist when writing or polishing marimo notebooks for
this repo.

## Notebook Shape

* A marimo notebook is a Python file whose cells are decorated with `@app.cell`.
  marimo derives dependencies from the global names each cell defines and
  references; serialized cell parameters and returns reflect those
  dependencies.
* Add PEP 723 metadata with `requires-python` and every runtime package the
  notebook imports, using repository-approved minimum version constraints such
  as `"marimo>=0.9"` and `"wandb>=0.18"`.
* Use one setup/import cell for shared imports, true constants, and environment
  detection.
* Keep notebook globals scarce. Use underscore-prefixed variables for simple
  cell-local temporaries and helper functions when a cell has substantial
  scratch logic.

## Reactivity

* Let the dependency graph determine execution. A cell runs when its inputs are
  ready.
* Do not rely on cross-cell mutation for reactivity; marimo does not track
  object mutations or attribute assignments. Prefer creating a new value, or
  mutate an object only in the cell that defines it.
* Avoid `mo.state()` unless bidirectional UI sync or accumulated callback state
  is required. Prefer ordinary variables and widget `.value` for normal
  notebook flow.

## Gating Expensive or Side-Effecting Work

Gate each expensive or externally side-effecting workflow stage once at its
boundary.

When the stage has configuration inputs, batch them into one form:

```python
form = mo.md("{epochs} {batch_size}").batch(
    epochs=epochs,
    batch_size=batch_size,
).form(submit_button_label="Train model", bordered=False)
form
```

```python
mo.stop(
    form.value is None,
    mo.md("Fill in the form above and click **Train model** to ..."),
)
cfg = form.value
```

When the stage has no configuration inputs, use `mo.ui.run_button` with
`mo.stop` instead of creating an empty form:

```python
run_step = mo.ui.run_button(label="Run step")
run_step
```

```python
mo.stop(
    not run_step.value,
    mo.md("Click **Run step** to continue."),
)

# Guarded implementation follows.
```

Keep the guarded implementation in the dependent code cell rather than hiding
it in a button or form callback.

Downstream cells should depend on post-gate names such as `cfg`, `model`, or
`results`. Do not re-check the same form or button in downstream cells or wrap
them in repeated `if` guards.

## Rendering and Presentation

* The final expression of a cell is what renders.
* Indented expressions inside `if`, `for`, `with`, or helper blocks do not
  become the cell output. Assign the display object, then put it last.
* Use markdown cells for prose and view cells for rendering.
* Keep view cells focused on presentation; move non-teaching heavy logic into
  named helpers.
* Prefer native components such as `mo.ui.table`, `mo.callout`, `mo.vstack`,
  and `mo.hstack` over formatting complex UI as markdown.

## UI

* Show widgets directly and read their `.value` from dependent cells.
* Prefer native `mo.ui` components before reaching for `anywidget`.

## Error Handling

* Do not use `try`/`except` for normal control flow.
* Let unexpected programming errors surface.
* Catch only specific, expected failures when the notebook can provide useful
  recovery guidance.

## Further Reference

For marimo behavior not covered here, prefer the
[official marimo documentation](https://docs.marimo.io/). Use upstream source
code only when the documented behavior is insufficient.
