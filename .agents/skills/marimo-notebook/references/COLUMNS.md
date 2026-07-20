A user may specify that they want to have a notebook with multiple columns. Below is an example of a notebook that does just that.

```python
# /// script
# dependencies = ["marimo"]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Column 1: Cool stuff

    This is where the user will first look. Put plots/inputs here typically.
    """)
    return


@app.cell
def _():
    # This cell is in column 1
    return


@app.cell
def _(mo):
    # This cell is in column 1 as well
    mo.ui.slider(1, 10, 1)
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Column 2: Boilerplate
    """)
    return


@app.cell
def _():
    # This cell is in column 2 
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
```

Notice the `@app.cell(column=0)` decorator? Every cell that follows sits in that column. Then, when we see `@app.cell(column=1)` the cells no longer fall into column 0 but they go into column 1.
