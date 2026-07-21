# Reactivity

## The DAG

marimo statically analyzes each cell to build a directed acyclic graph:

- **References** = global variables the cell reads (function parameters)
- **Definitions** = global variables the cell creates (return tuple)

When a cell runs, all cells that reference its definitions automatically run. Execution order is determined by the DAG, not cell position on the page.

## Variable Uniqueness

Every global variable must be defined by exactly one cell. This prevents ambiguity in the dependency graph.

If you need the same name in multiple cells, use underscore-prefixed cell-local variables:

```python
@app.cell
def _():
    _temp = expensive_computation()
    result_a = summarize(_temp)
    return (result_a,)

@app.cell
def _():
    _temp = different_computation()  # no conflict, _temp is cell-local
    result_b = summarize(_temp)
    return (result_b,)
```

## Mutations Are Not Tracked

marimo does **not** detect mutations like `.append()`, attribute assignment, or in-place DataFrame operations across cells.

```python
# BAD — mutation in another cell, marimo won't re-run dependents
# Cell 1
items = [1, 2, 3]
# Cell 2
items.append(4)  # invisible to the DAG

# GOOD — create a new variable
# Cell 2
extended = items + [4]
```

Mutations within the same cell that defines the variable are fine:

```python
@app.cell
def _(pd):
    df = pd.DataFrame({"a": [1, 2]})
    df["b"] = [3, 4]  # same cell, fine
    return (df,)
```

## Deleting Cells

Deleting a cell removes its global variables from memory. Cells that referenced those variables become invalidated.

## Disabling Cells

Disable a cell to prevent it and its dependents from running. Re-enabling triggers a re-run if upstream cells changed while it was disabled.

## Lazy Evaluation

Instead of auto-running dependents, mark them stale for manual execution. Configure in runtime settings or use `mo.lazy()` for specific elements.
