# Marimo Idioms

Use this as a short review checklist when writing or polishing marimo notebooks
for this repo.

## Notebook Shape

- A marimo notebook is a Python file. Cells are functions decorated with
  `@app.cell`; dependencies are the function arguments and return values.
- Use a single setup cell for imports, constants, and environment detection.
- Add PEP 723 script metadata at the top so `uvx marimo ... --sandbox` can
  recreate the runtime environment.
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

## Rendering

- The final expression of a cell is what renders.
- Indented expressions inside `if`, `for`, `with`, or helper blocks do not
  become the cell output. Assign the display object, then put it last.
- Use markdown cells for prose. Use view cells for rendering. Keep heavy logic
  in named helpers.

## UI

- Prefer a single submittable form for controls that trigger expensive work.
- Show widgets directly; downstream cells should read `.value`.
- Prefer native `mo.ui` components before reaching for anywidget.
- Use real marimo display components, such as `mo.ui.table`, `mo.callout`,
  `mo.vstack`, and `mo.hstack`, instead of formatting complex UI as markdown.

## Error Handling

- Do not use `try`/`except` for normal control flow.
- Let unexpected programming errors surface.
- Catch only specific, expected failures where the notebook can give useful
  recovery guidance, such as W&B auth or account setup problems.

## Verification

Run this before handing back:

```bash
uvx marimo check <notebook.py>
```
