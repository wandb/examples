# Convert Cleanup

Use this after `scripts/convert-colab-to-marimo.py` creates the initial
marimo notebook from a Jupyter `.ipynb`. The converter writes diagnostics to
`marimo/convert/<name>/.logs/`.

## Start from convert logs

- For batch runs, start with `marimo/convert/convert-summary.txt` to find
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

- Follow [`tutorial-notebook-objectives.md`](tutorial-notebook-objectives.md)
  when deciding which instructional cells and W&B API examples to preserve.
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

## Common Check Failures

- Circular dependencies often come from imports or helper names returned by a
  later cell and consumed by an earlier helper cell. Fix by moving shared
  imports/constants into `app.setup(...)` or into the helper cell that uses
  them, and avoid returning imported symbols from downstream logic cells.
- `multiple-definitions` errors can happen after moving the same import into
  multiple cells. Keep cell-local imports private with an underscore alias
  such as `from torch.utils.data import DataLoader as _DataLoader`.

## Molab and Remote Assets

- Do not assume molab's **Mirror from GitHub** action provides a checkout of the
  whole repository. A notebook and its dependency metadata can be present
  while sibling data files are not.
- For public, read-only repository assets, prefer a named HTTP filesystem and
  raw content URLs, for example `repo_fs = fsspec.filesystem("https")` with a
  `raw.githubusercontent.com` base URL. A public filesystem name also makes the
  source discoverable in marimo's Remote Storage panel.
- Avoid an anonymous `fsspec.filesystem("github", ...)` in hosted notebooks.
  Its repository lookup uses the rate-limited GitHub API, and users can share
  one unauthenticated IP quota. Use it only when repository listing semantics
  are required and an optional GitHub token comes from the runtime's secrets or
  environment; never hardcode the token.
- Declare `fsspec[http]`, not only `fsspec`, for HTTP files, Git-LFS content,
  and GitHub files larger than 1 MB.
- Pass file-like objects directly when the consumer supports them. Otherwise,
  adapt in memory with `io.BytesIO` or `io.StringIO`, or materialize only the
  specific file an API requires.
- Clone a repository only when the tutorial needs Git history, repository
  semantics, or a local directory tree rather than a few read-only assets.

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

After the final notebook edit, run:

```bash
uvx marimo check marimo/convert/<example-name>/<example_name>.py
```

Then open the notebook in a fresh molab session or a local sandbox. Confirm
that intended controls and embeds render, change each safe control at least
once, and inspect cell errors. Static checking cannot detect a nonexistent
runtime attribute such as `.clicked` or a widget that was constructed but
never returned for display. Keep remote-write controls unsubmitted, or use an
offline/test backend, during this smoke test.
