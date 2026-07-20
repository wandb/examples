---
name: marimo-notebook
description: Write a marimo notebook in a Python file in the right format.
---

# Notes for marimo Notebooks

marimo uses Python to create notebooks, unlike Jupyter which uses JSON. Here's an example notebook: 

```python
# /// script
# dependencies = [
#     "marimo",
#     "numpy==2.4.3",
# ]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell
def _():
    print("hello world")
    return


@app.cell
def _(np, slider):
    np.array([1,2,3]) + slider.value
    return


@app.cell
def _(mo):
    slider = mo.ui.slider(1, 10, 1, label="number to add")
    slider
    return (slider,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

```

Notice how the notebook is structured with functions can represent cell contents. Each cell is defined with the `@app.cell` decorator and the inputs/outputs of the function are the inputs/outputs of the cell. marimo usually takes care of the dependencies between cells automatically. 

## Running Marimo Notebooks

```bash
# Run as script (non-interactive, for testing)
uv run <notebook.py>

# Run interactively in browser
uv run marimo run <notebook.py>

# Edit interactively
uv run marimo edit <notebook.py>
```

## Script Mode Detection

Use `mo.app_meta().mode == "script"` to detect CLI vs interactive:

```python
@app.cell
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)
```

## Key Principle: Keep It Simple

**Show all UI elements always.** Only change the data source in script mode.

- Sliders, buttons, widgets should always be created and displayed
- In script mode, just use synthetic/default data instead of waiting for user input
- Don't wrap everything in `if not is_script_mode` conditionals
- Don't use try/except for normal control flow

### Good Pattern

```python
# Always show the widget
@app.cell
def _(ScatterWidget, mo):
    scatter_widget = mo.ui.anywidget(ScatterWidget())
    scatter_widget
    return (scatter_widget,)

# Only change data source based on mode
@app.cell
def _(is_script_mode, make_moons, scatter_widget, np, torch):
    if is_script_mode:
        # Use synthetic data for testing
        X, y = make_moons(n_samples=200, noise=0.2)
        X_data = torch.tensor(X, dtype=torch.float32)
        y_data = torch.tensor(y)
        data_error = None
    else:
        # Use widget data in interactive mode
        X, y = scatter_widget.widget.data_as_X_y
        # ... process data ...
    return X_data, y_data, data_error

# Always show sliders - use their .value in both modes
@app.cell
def _(mo):
    lr_slider = mo.ui.slider(start=0.001, stop=0.1, value=0.01)
    lr_slider
    return (lr_slider,)

# Auto-run in script mode, wait for button in interactive
@app.cell
def _(is_script_mode, train_button, lr_slider, run_training, X_data, y_data):
    if is_script_mode:
        # Auto-run with slider defaults
        results = run_training(X_data, y_data, lr=lr_slider.value)
    else:
        # Wait for button click
        if train_button.value:
            results = run_training(X_data, y_data, lr=lr_slider.value)
    return (results,)
```

## State and Reactivity

Variables between cells define the reactivity of the notebook for 99% of the use-cases out there. No special state management needed. Don't mutate objects across cells (e.g., `my_list.append()`); create new objects instead. Avoid `mo.state()` unless you need bidirectional UI sync or accumulated callback state. See [STATE.md](references/STATE.md) for details.

## Don't Guard Cells with `if` Statements

Marimo's reactivity means cells only run when their dependencies are ready. Don't add unnecessary guards:

```python
# BAD - the if statement prevents the chart from showing
@app.cell
def _(plt, training_results):
    if training_results:  # WRONG - don't do this
        fig, ax = plt.subplots()
        ax.plot(training_results['losses'])
        fig
    return

# GOOD - let marimo handle the dependency
@app.cell
def _(plt, training_results):
    fig, ax = plt.subplots()
    ax.plot(training_results['losses'])
    fig
    return
```

The cell won't run until `training_results` has a value anyway.

## Don't Use try/except for Control Flow

Don't wrap code in try/except blocks unless you're handling a specific, expected exception. Let errors surface naturally.

```python
# BAD - hiding errors behind try/except
@app.cell
def _(scatter_widget, np, torch):
    try:
        X, y = scatter_widget.widget.data_as_X_y
        X = np.array(X, dtype=np.float32)
        # ...
    except Exception as e:
        return None, None, f"Error: {e}"

# GOOD - let it fail if something is wrong
@app.cell
def _(scatter_widget, np, torch):
    X, y = scatter_widget.widget.data_as_X_y
    X = np.array(X, dtype=np.float32)
    # ...
```

Only use try/except when:
- You're handling a specific, known exception type
- The exception is expected in normal operation (e.g., file not found)
- You have a meaningful recovery action

## Cell Output Rendering

Marimo only renders the **final expression** of a cell. Indented or conditional expressions won't render:

```python
# BAD - indented expression won't render
@app.cell
def _(mo, condition):
    if condition:
        mo.md("This won't show!")  # WRONG - indented
    return

# GOOD - final expression renders
@app.cell
def _(mo, condition):
    result = mo.md("Shown!") if condition else mo.md("Also shown!")
    result  # This renders because it's the final expression
    return
```

## PEP 723 Dependencies

Notebooks created via `marimo edit --sandbox` have these dependencies added to the top of the file automatically but it is a good practice to make sure these exist when creating a notebook too: 

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "torch>=2.0.0",
# ]
# ///
```

## marimo check 

When working on a notebook it is important to check if the notebook can run. That's why marimo provides a `check` command that acts as a linter to find common mistakes. 

```bash
uvx marimo check <notebook.py>
```

Make sure these are checked before handing a notebook back to the user. 

**Important**: you have a tendency to over-do variables with an underscore prefix. You should only apply this to one or two variables at most. Consider creating a new variable instead of prefixing entire cells in marimo. 

## api docs

If the user specifically wants you to use a marimo function, you can locally check the docs via: 

```
uv --with marimo run python -c "import marimo as mo; help(mo.ui.form)"
```

## tests 

By default, marimo discovers and executes tests inside your notebook.
When the optional `pytest` dependency is present, marimo runs `pytest` on cells that
consist exclusively of test code - i.e. functions whose names start with `test_`. 
If the user asks you to add tests, make sure to add the `pytest` dependency is added and that
there is a cell that contains only test code.

For more information on testing with pytest see [PYTEST.md](references/PYTEST.md)

Once tests are added, you can run pytest from the commandline on the notebook to run pytest. 

```
pytest <notebook.py>
```

## Additional resources

- For marimo notebooks that run in width=columns [SQL.md](references/COLUMNS.md)
- For SQL use in marimo see [SQL.md](references/SQL.md)
- For UI elements in marimo [UI.md](references/UI.md)
- For exposing functions/classes as top level imports [TOP-LEVEL-IMPORTS.md](references/TOP-LEVEL-IMPORTS.md)
- For exporting notebooks (PDF, HTML, markdown, etc.) [EXPORTS.md](references/EXPORTS.md)
- For state management and reactivity [STATE.md](references/STATE.md)
- For deployment of marimo notebooks [DEPLOYMENT.md](references/DEPLOYMENT.md)
- For custom interactive widgets with anywidget [ANYWIDGET.md](references/ANYWIDGET.md)
- For external editing and `--watch` mode [WATCHING.md](references/WATCHING.md)
- For expensive notebooks (caching, lazy eval, mo.stop) [EXPENSIVE.md](references/EXPENSIVE.md)
- For configuration (pyproject.toml, marimo.toml) [CONFIGURATION.md](references/CONFIGURATION.md)
- For reactivity model (DAG, variable scoping, mutations) [REACTIVITY.md](references/REACTIVITY.md)
