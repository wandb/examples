---
name: jupyter-to-marimo
description: Convert a Jupyter notebook (.ipynb) to a marimo notebook (.py).
---

# Converting Jupyter Notebooks to Marimo

**IMPORTANT**: When asked to translate a notebook, ALWAYS run `uvx marimo convert <notebook.ipynb> -o <notebook.py>` FIRST before reading any files. This saves precious tokens - reading large notebooks can consume 30k+ tokens, while the converted .py file is much smaller and easier to work with.

## Steps

1. **Convert using the CLI**

Run the marimo convert command via `uvx` so no install is needed:

```bash
uvx marimo convert <notebook.ipynb> -o <notebook.py>
```

This generates a marimo-compatible `.py` file from the Jupyter notebook.

2. **Run `marimo check` on the output**

```bash
uvx marimo check <notebook.py>
```

Fix any issues that are reported before continuing.

3. **Review and clean up the converted notebook**

Read the generated `.py` file and apply the following improvements:

- Ensure the script metadata block lists all required packages. The converter may miss some.
- Drop leftover Jupyter artifacts like `display()` calls, or `%magic` commands that don't apply in marimo.
- Make sure the final expression of each cell is the value to render. Indented or conditional expressions won't display.
- If the original notebook requires environment variables via an input, consider adding the `EnvConfig` widget from wigglystuff. Details can be found [here](https://koaning.github.io/wigglystuff/reference/env-config.md).
- If the original notebook uses ipywidgets, see `references/widgets.md` for a full mapping of ipywidgets to marimo equivalents, including patterns for callbacks, linking, and anywidget integration.
- If the notebook contains LaTeX, see `references/latex.md` for how to port MathJax syntax to KaTeX (which marimo uses).

4. **Run `marimo check` again** after your edits to confirm nothing was broken.

