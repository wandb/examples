You can import top-level functions and classes defined in a marimo notebook into other Python scripts or notebooks using normal Python syntax, as long as your definitions satisfy the simple criteria described on this page. This makes your notebook code reusable, testable, and easier to edit in text editors of your choice.

For a function or class to be saved at the top level of the notebook file, it must meet the following criteria:

The cell must define just a single function or class.
The defined function or class can only refer to symbols defined in the setup cell, or to other top-level symbols.

```python
# /// script
# dependencies = [
#     "marimo",
#     "numpy==2.4.2",
# ]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")

# Define setup cell
with app.setup:
    import numpy as np


# Define function cell
@app.function
def calculate_statistics(data):
    """Calculate basic statistics for a dataset"""
    return {
        "mean": np.mean(data),
        "median": np.median(data),
        "std": np.std(data)
    }


@app.cell
def _():
    import marimo as mo

    return

if __name__ == "__main__":
    app.run()
```

In this example, the setup cell is represented as a context manager `app.setup` and the cell that contains `calculate_statistics` is represented as a function decorator `@app.function`. You can now import `calculate_statistics` from other Python scripts or notebooks. There can be no more than one setup cell per notebook.

```python
# In another_script.py
from my_notebook import calculate_statistics

data = [1, 2, 3, 4, 5]
stats = calculate_statistics(data)
print(stats)
```