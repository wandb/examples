# Expensive Notebooks

## mo.stop()

Halt cell execution when a condition is met:

```python
@app.cell
def _(mo, data):
    mo.stop(data is None, mo.md("Waiting for data..."))
    # Only runs if data is not None
    result = process(data)
    return (result,)
```

Pair with `mo.ui.run_button()` for manual triggers:

```python
@app.cell
def _(mo):
    run_btn = mo.ui.run_button(label="Run analysis")
    run_btn
    return (run_btn,)

@app.cell
def _(mo, run_btn, data):
    mo.stop(not run_btn.value)
    result = expensive_analysis(data)
    return (result,)
```

## mo.cache

In-memory cache for the current session. Results are reused when inputs match:

```python
@mo.cache
def fetch_data(query: str):
    return db.execute(query)
```

Works as a decorator or context manager.

## mo.persistent_cache

Disk-based cache that persists across notebook restarts:

```python
@mo.persistent_cache
def train_model(params):
    return heavy_training(params)
```

## mo.lazy()

Defer rendering and computation until needed:

```python
# Only render table when it scrolls into view
mo.lazy(mo.ui.table(large_df))

# Only compute when tab is selected
mo.ui.tabs({
    "Summary": summary,
    "Details": mo.lazy(lambda: expensive_query()),
})
```

## Runtime Configuration

- Disable autorun on cell changes for long-running notebooks
- Disable startup autorun to prevent automatic execution on open
- Disable individual cells temporarily during editing

## Memory Management

Wrap intermediate computations in functions so local variables get freed:

```python
@app.cell
def _():
    def _compute():
        large_data = load_everything()
        result = summarize(large_data)
        return result  # large_data is freed here
    output = _compute()
    return (output,)
```
