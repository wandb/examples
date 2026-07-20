# State in marimo

## Reactivity IS State Management

In marimo, regular Python variables between cells are your state. When a cell assigns a variable, all cells that read it re-run automatically. Widget values (`widget.value`) work the same way — interact with a widget and dependent cells re-execute. No store, no session_state, no hooks needed.

## Don't Mutate Objects Across Cells

marimo does **not** track mutations like `my_list.append(42)` or `obj.value = 42`.

```python
# BAD - mutation in another cell won't trigger re-runs
# Cell 1
items = [1, 2, 3]

# Cell 2
items.append(4)  # marimo won't know this happened

# GOOD - create new objects instead
# Cell 1
items = [1, 2, 3]

# Cell 2
extended_items = items + [4]
```

## You Probably Don't Need `mo.state()`

In 99% of cases, built-in reactivity is enough:

- **Reading widget values** — just use `widget.value` in another cell
- **Combining multiple inputs** — use `.batch().form()`
- **Conditional data** — use `if`/`else` in one cell

## When You Do Need `mo.state()`

Use it when you need **accumulated state from callbacks** or **bidirectional sync** between UI elements.

```python
get_val, set_val = mo.state(initial_value)
```

- Read: `get_val()`
- Update: `set_val(new_value)` or `set_val(lambda d: d + [new_item])`
- The cell calling the setter does NOT re-run (unless `allow_self_loops=True`)

### Example: todo list with accumulated state

```python
# Cell 1 — declare state
@app.cell
def _(mo):
    get_items, set_items = mo.state([])
    return get_items, set_items

# Cell 2 — input form
@app.cell
def _(mo, set_items):
    task = mo.ui.text(label="New task")
    add = mo.ui.button(
        label="Add",
        on_click=lambda _: set_items(lambda d: d + [task.value])
    )
    mo.hstack([task, add])
    return

# Cell 3 — display (re-runs when state changes)
@app.cell
def _(mo, get_items):
    mo.md("\n".join(f"- {t}" for t in get_items()))
    return
```

### Example: syncing two UI elements

```python
@app.cell
def _(mo):
    get_n, set_n = mo.state(50)
    return get_n, set_n

@app.cell
def _(mo, get_n, set_n):
    slider = mo.ui.slider(0, 100, value=get_n(), on_change=set_n)
    number = mo.ui.number(0, 100, value=get_n(), on_change=set_n)
    mo.hstack([slider, number])
    return
```

## Warnings

- Don't store `mo.ui` elements inside state — causes hard-to-diagnose bugs.
- Don't use `on_change` when you can just read `.value` from another cell.
- Write idempotent cells — same inputs should produce same outputs.
