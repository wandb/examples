# Marimo Idioms

Use this as a review checklist when writing or polishing marimo notebooks for
this repo.

## Notebook Shape

- A marimo notebook is a Python file whose cells are decorated with `@app.cell`.
  marimo derives dependencies from the global names each cell defines and
  references; serialized cell parameters and returns reflect those
  dependencies.
- Add PEP 723 metadata with `requires-python` and every runtime package the
  notebook imports, using repository-approved minimum version constraints such
  as `"marimo>=0.9"` and `"wandb>=0.18"`.
- Use one setup/import cell for shared imports, true constants, and environment
  detection.
- Keep reactive notebook globals scarce.

## Separate Teaching, Orchestration, and Helpers

Distinguish these roles when converting a tutorial:

- Teaching code: Code the reader is meant to learn from. Preserve a clean,
  normal-Python shape similar to the source Colab notebook.
- Marimo orchestration: Forms, buttons, `mo.stop`, widget `.value`, and
  reactive wiring. Keep these in small `@app.cell` cells around the teaching
  code.
- Reusable helpers: Procedural code that is clearer as a named function.
  Prefer `@app.function` for reusable top-level helpers and pass reactive values
  as function arguments.

Do not add marimo orchestration, generated dependency plumbing, or
underscore-prefixed scratch variables to teaching code merely to satisfy the
reactive graph.

Preserve original identifiers in teaching code. Do not mechanically prefix a
unique name with `_` merely because no later cell reads it. First inspect actual
definitions and references across cells. For reader-facing definitions, resolve
a real cross-cell redefinition by moving procedural work into a function or
choosing unique descriptive names. Reserve private names for newly introduced
implementation-only plumbing, such as file handles, context-manager targets, or
UI internals. For W&B run objects, follow
[`Naming Run Objects in marimo`](wandb-patterns.md#naming-run-objects-in-marimo).

When teaching code is naturally expressed as a function, keep the function
clean and put its gate or UI wiring in separate cells.

## Reactivity

- Let the dependency graph determine execution. A cell runs when its inputs are
  ready.
- `mo.stop()`, `if`, `for`, and `with` control runtime execution; they do not
  create a static scope. Imports, assignment targets, loop targets, and context
  manager targets anywhere in a cell still define names in marimo's graph. Give
  each shared name one owning cell. Keep unique teaching names public even when
  they have no downstream consumer; use `_` for genuinely private plumbing or
  repeated scratch names that are not reader-facing, or move procedural work
  into a function.
- Do not rely on cross-cell mutation for reactivity; marimo does not track
  object mutations or attribute assignments. Prefer creating a new value, or
  mutate an object only in the cell that defines it.
- Avoid `mo.state()` unless bidirectional UI sync or accumulated callback state
  is required. Prefer ordinary variables and widget `.value` for normal
  notebook flow.

## Gating Expensive or Side-Effecting Work

Gate each expensive or externally side-effecting workflow stage once at its
boundary.

Treat an explicit form submission or run-button click as the reader's consent
boundary for remote writes. Label the control with the action it performs and
state what it creates. Opening the notebook, changing an unsubmitted control,
or lazily rendering content must not create a remote object; `mo.lazy()` is not
a substitute for explicit consent.

Keep forms, buttons, widget `.value`, and `mo.stop(...)` in small orchestration
cells rather than mixing them into teaching code.

When a stage has configuration inputs, batch them into a form. When it has no
configuration inputs, use `mo.ui.run_button` with `mo.stop` instead of creating
an empty form.

Prefer this separation:

```python
@app.cell(hide_code=True)
def _(form):
    mo.stop(
        form.value is None,
        mo.md("Fill in the form above and click Train model to continue."),
    )
    config = form.value
    return (config,)
```

```python
@app.function
def train_model(config):
    # Clean tutorial implementation.
    ...
    return model
```

```python
@app.cell
def _(config):
    model = train_model(config)
    return (model,)
```

Pass post-gate values into clean teaching code or named `@app.function`
helpers. Do not repeat the same gate in downstream cells.

## Rendering and Presentation

- The final expression of a cell is what renders.
- Indented expressions inside `if`, `for`, `with`, or helper blocks do not
  become the cell output. Assign the display object, then put it last.
- Use markdown cells for prose and view cells for rendering.
- Keep view cells focused on presentation; move non-teaching heavy logic into
  named helpers.
- Preserve deliberate `hide_code` choices. Prefer `hide_code=True` for
  implementation-only cells whose rendered output is the reader-facing
  surface, such as authentication form construction, W&B connection or status
  gates, and boilerplate HTML embeds such as YouTube iframes. Keep teaching
  code, featured W&B SDK usage, and helper implementations readers are expected
  to adapt visible.
- Prefer native components such as `mo.ui.table`, `mo.callout`, `mo.vstack`,
  and `mo.hstack` over formatting complex UI as markdown.
- Use `mo.video` for a direct video URL, file, or bytes. For a hosted player
  such as YouTube, use the provider's canonical HTTPS embed URL in a trusted
  `mo.Html` iframe with a descriptive title and a normal link fallback.
- For the W&B header pattern used by the media tutorial, use
  `https://raw.githubusercontent.com/wandb/docs/main/icons/Endorsed_primary_blackwhite.svg`
  in the light theme and
  `https://raw.githubusercontent.com/wandb/docs/main/icons/Endorsed_primary_goldwhite.svg`
  in the dark theme. Render both and switch them with marimo's `body.dark`
  class; visually verify both themes. For the verified marimo `mo.callout`
  pattern, select the theme with `:host-context(body.dark)` so the rule crosses
  the component boundary.

## UI

- Constructing or assigning a widget does not display it. End the definition
  cell with the widget or a layout containing it; returning it only wires the
  reactive dependency graph.
- Show widgets directly and read documented reactive state such as `.value` in
  orchestration cells. Do not invent callback-style attributes such as
  `.clicked`; inspect the live object or official API when uncertain.
- Prefer native `mo.ui` components before reaching for `anywidget`.

## Error Handling

- Do not use `try`/`except` for normal control flow.
- Let unexpected programming errors surface.
- Catch only specific, expected failures when the notebook can provide useful
  recovery guidance.

## Further Reference

For marimo behavior not covered here, prefer the
[official marimo documentation](https://docs.marimo.io/). Use upstream source
code only when the documented behavior is insufficient.
