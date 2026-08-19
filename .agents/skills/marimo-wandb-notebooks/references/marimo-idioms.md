# Marimo Idioms

Use this as a short review checklist when writing or polishing marimo notebooks
for this repo.

## Marimo Notebook Shape

- A marimo notebook is a Python file whose cells are decorated with
  `@app.cell`. marimo derives dependencies from the global names each cell
  defines and references; the serialized cell parameters and returns reflect
  those dependencies.
- Add PEP 723 metadata with `requires-python` and every runtime package the
  notebook imports, using repository-approved minimum version constraints such
  as `"marimo>=0.9"` and `"wandb>=0.18"`.
- Use the marimo setup cell for shared imports, true constants, and environment
  detection. Keep tutorial parameters and reactive values in regular cells.
- Keep notebook globals scarce. Use underscore-prefixed variables for
  cell-local temporaries; use helper functions when a cell has substantial scratch logic.

## Reactivity

- Let the dependency graph do the work. A cell runs when its inputs are ready.
- Do not mutate shared objects across cells, such as `items.append(...)` or
  in-place DataFrame edits. Create a new value instead.
- Avoid `mo.state()` unless you need bidirectional UI sync or accumulated
  callback state. Most notebooks only need ordinary variables and `.value`.
- Do not wrap downstream cells in repeated `if form.value` or button checks.
  Gate once with `mo.stop`, then make later cells depend on names defined after
  the gate.
- Gate expensive or externally side-effecting operations so ordinary reactive
  updates do not repeat them. Examples include model training, W&B run
  creation, artifact uploads, and Registry mutations.

## Gating Expensive Work

For each expensive workflow stage, batch its controls into one form and gate
once at the stage boundary:

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

If an expensive action has no configuration inputs, prefer `mo.ui.run_button`
with `mo.stop` instead of creating an empty form.

Downstream cells should depend on post-gate names such as `cfg`, `run`, or
`model`. Do not re-check the form in later cells or wrap cells in `if` guards.

## Rendering

- The final expression of a cell is what renders.
- Indented expressions inside `if`, `for`, `with`, or helper blocks do not
  become the cell output. Assign the display object, then put it last.
- Use markdown cells for prose. Use view cells for rendering. Keep
  non-teaching heavy logic in named helpers.

## Logic And Presentation

- View cells should keep rendering code short and avoid unrelated computation.
- Push temporaries into functions to keep notebook globals to a minimum. Every
  returned name is reserved across the whole file.
- Present results with real components, such as `mo.ui.table`,
  `mo.callout(kind="success")`, `mo.vstack`, and `mo.hstack`, instead of
  formatting complex UI as markdown.

## UI

- Prefer a single submittable form for controls that trigger expensive work.
- Show widgets directly; downstream cells should read `.value`.
- Prefer native `mo.ui` components before reaching for anywidget.

## Error Handling

- Do not use `try`/`except` for normal control flow.
- Let unexpected programming errors surface.
- Catch only specific, expected failures where the notebook can give useful
  recovery guidance, such as W&B auth or account setup problems.
