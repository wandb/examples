# Conversion Cleanup

Use this after `scripts/convert-colab-to-marimo.py` creates the initial
marimo notebook from a Jupyter `.ipynb`. The converter writes diagnostics to
`marimo/convert/<name>/.logs/`.

## Start from conversion logs

- For batch runs, start with `marimo/convert/conversion-summary.txt` to find
  notebooks that need action.
- Read `marimo/convert/<name>/.logs/result.json` first. Check `status`,
  `failed_stage`, `source`, `target`, and each command's exit code.
- If `failed_stage` is `convert`, read
  `marimo/convert/<name>/.logs/marimo-convert.log` before editing the notebook.
- If `failed_stage` is `check`, read
  `marimo/convert/<name>/.logs/marimo-check.log` before editing the notebook.
- If `status` is `ok`, still skim `result.json` to confirm the source and target
  paths before cleanup.
- Fix `marimo check` issues first; they often point to converted cells that
  need to be split, reordered, or moved into helpers.

## Common Converter Leftovers

- Ensure the PEP 723 script metadata lists every runtime package the notebook
  imports. The converter may miss dependencies.
- Remove Jupyter-only artifacts such as `%magic` commands, shell escapes, and
  unnecessary `display()` calls.
- Make the intended output the final expression of each display cell. Indented
  or conditional expressions will not render as cell output.
- Replace notebook-global scratch variables with local variables inside helper
  functions when they are only used in one step.
- Prefer explicit markdown cells for prose. Do not leave tutorial text inside
  code comments or string literals in logic cells.

## Widget Cleanup

- Replace ipywidgets with native `mo.ui` components when there is a direct
  equivalent.
- Replace `interact`, `observe`, `link`, and `jslink` patterns with marimo
  reactivity. Split UI definition, value consumption, and rendering into
  separate cells.
- Use `mo.ui.anywidget()` only when no native marimo component fits.

## LaTeX Cleanup

- Use raw strings for markdown containing LaTeX, such as `mo.md(r"$x^2$")`.
- Replace MathJax-only constructs with KaTeX-compatible syntax.
- Visually verify math-heavy outputs because KaTeX failures can be quiet.

## Final Check

Run:

```bash
uvx marimo check examples/marimo/<example-name>/<example_name>.py
```
