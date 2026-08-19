# Marimo Idioms

Use this as a short review checklist when writing or polishing marimo notebooks
for this repo.

## Marimo Notebook Shape

- A marimo notebook is a Python file. Cells are functions decorated with
  `@app.cell`; dependencies are the function arguments and return values.
- Use a single setup cell for imports, constants, and environment detection.
- Keep globals scarce. Every returned name is notebook-wide, so move
  step-local scratch work into helper functions.

## Reactivity

- Let the dependency graph do the work. A cell runs when its inputs are ready.
- Do not mutate shared objects across cells, such as `items.append(...)` or
  in-place DataFrame edits. Create a new value instead.
- Avoid `mo.state()` unless you need bidirectional UI sync or accumulated
  callback state. Most notebooks only need ordinary variables and `.value`.
- Do not wrap downstream cells in repeated `if form.value` or button checks.
  Gate once with `mo.stop`, then make later cells depend on names defined after
  the gate.

## Gating Expensive Work

Batch controls into one form, then gate exactly once:

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

Downstream cells should depend on post-gate names such as `cfg`, `run`, or
`model`. Do not re-check the form in later cells or wrap cells in `if` guards.

Label the submit button with the remote action, such as **Start run** or
**Train model**, and state what the action creates. Treat submission as the
consent boundary for remote writes; do not use `mo.lazy()` for this purpose.

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
