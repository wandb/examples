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
- Move `%pip` and `!pip install` requirements into PEP 723 metadata, then remove
  the obsolete install command, generated install commentary, and any empty
  cell it leaves behind. Do not replace package installation with a subprocess.
- Remove Jupyter-only artifacts such as `%magic` commands, shell escapes, and
  unnecessary `display()` calls.
- Translate remaining shell escapes by purpose instead of mechanically wrapping
  every command in `subprocess`. Use `fsspec` when file-like or filesystem
  access is useful, and reserve `subprocess` for programs that genuinely need a
  separate process.
- Make the intended output the final expression of each display cell. Indented
  or conditional expressions will not render as cell output.
- When cleanup is requested and it improves clarity, replace notebook-global
  scratch variables with locals inside helper functions. Do not do this during
  an exact synchronization or when it would rename or obscure teaching code
  without resolving a demonstrated graph problem.
- Use explicit markdown cells for newly added narrative. Preserve source code
  comments and docstrings in place; do not move or rewrite them as prose.

## Common Check Failures

- Circular dependencies often come from imports or helper names returned by a
  later cell and consumed by an earlier helper cell. Fix by moving shared
  imports into setup or the helper that uses them, and avoid returning imported
  symbols from downstream logic cells. Pass educational configuration as
  arguments instead of hoisting it away from its teaching step.
- `multiple-definitions` errors can happen after moving the same import into
  multiple cells. Keep cell-local imports private with an underscore alias
  such as `from torch.utils.data import DataLoader as _DataLoader`.

## Notebook Links and HTML Markers

Classify links before removing them:

- A Colab badge linking to the source of this same notebook should become one
  **Open in molab** badge targeting its converted `.py`, near the top. Remove
  duplicate "original Colab tutorial" links rather than keeping both versions.
- A link to a different tutorial in this repository should target that
  tutorial's converted notebook when available. Preserve the useful cross-link
  and surrounding explanation. Resolve shortened URLs before mapping them;
  the label can be stale or the redirect can point back to the current notebook.
- External scientific attribution, upstream project references, and official
  documentation are distinct from obsolete self-links. Preserve them unless
  the user requests their removal. If a tutorial has no converted counterpart,
  report that gap rather than inventing a target.

Use the existing badge shape:

```markdown
[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/REF/marimo/convert/EXAMPLE/NOTEBOOK.py/server)
```

Substitute the real repository-relative path and requested Git ref. For testing
from a branch, verify the published remote branch and file; a local branch name
can differ from its upstream. When the user says to assume the PR is merged,
use `main`. Validate that the intended converted file exists and distinguish
links intended for after merge from links verified live today. Do not claim a new
`main` URL works before the file is merged.

Remove W&B HTML tracking comments such as `<!--- @wandbcode{alphafold} -->`,
including the `<!-- @wandbcode{...} -->` variant. If nothing remains in that
markdown cell, remove the whole cell. If a banner, logo, prose, or other content
remains, keep it. Do not confuse these markers with ordinary Python comments.

For repository-wide cleanup, inventory all marimo notebooks, not just the
current batch. Check for direct Colab URLs and short links such as
`wandb.me/sweeps-colab`, verify converted targets, and confirm each converted
tutorial has a self-link badge without duplicates. Keep generated `.md`,
`__marimo__` snapshots, and original `.ipynb` sources outside the edit scope
unless explicitly requested.

## Hosted Runtime and Accelerator Checks

Local success does not establish molab compatibility. When a hosted notebook
fails, inspect versions and imported module locations in its actual kernel,
including preinstalled packages and the Python version. Dependency-panel labels
or PEP 723 metadata alone do not prove which modules are currently loaded.

- For a PyTorch-only Transformers tutorial, an installed TensorFlow/Keras stack
  can trigger an unrelated optional-backend import failure. When applicable to
  the installed Transformers version, set `os.environ["USE_TF"] = "0"` before
  any Transformers import. Do not add `tf-keras` to a PyTorch lesson just because
  the optional backend's traceback recommends it. A previously imported module
  may cache availability; account for that when validating the repair.
- For ordinary PyTorch device selection, prefer CUDA when available, then Apple
  MPS, then CPU. MPS keeps the existing PyTorch model and training code; MLX is a
  different implementation, not a device switch. Validate the operations used
  by the notebook, including export when featured, on the backend being added.
- A pytest fixture error on a helper named `test` can be notebook test discovery,
  not a model evaluation failure. Inspect the classification and existing
  training result before retraining; see
  [`marimo-idioms.md`](marimo-idioms.md#separate-teaching-orchestration-and-helpers).

### Repairing a running notebook

When pairing on a live session, use the available `marimo-pair` skill and its
code-mode API for durable edits. Inspect current API help instead of assuming
private kernel attributes are stable. Use the supplied session credentials
without printing them or storing them in notebook code.

Before changing a helper, inspect its downstream cells and submitted controls.
In autorun mode, even a helper rename can rerun training or create another W&B
run. Prefer temporary lazy execution or another supported means of limiting
execution to the cells under repair, and restore the original execution mode
afterward. Preserve trained models, results, and unrelated user edits. Use
cached data and local capture/offline logging to validate a media or evaluation
repair when another remote run is unnecessary. Do not finish or restart the
user's active run solely to perform a read-only check.

When the request includes the local copy, apply the verified live fix locally
too and check both artifacts. A full requested sync should preserve exported
cell order and metadata; a narrow repair should not overwrite unrelated local
changes. Do not commit or push unless requested.

## Molab and Remote Assets

- Do not assume molab's **Mirror from GitHub** action provides a checkout of the
  whole repository. A notebook and its dependency metadata can be present
  while sibling data files are not.
- Preserve a working asset-loading backend. Do not replace it solely because a
  different backend might avoid a hypothetical hosted-environment limit.
- Use `fsspec.filesystem("github", org=..., repo=...)` with repository-relative
  paths when repository listing or marimo's Remote Storage browser is useful.
  Anonymous hosted sessions can share GitHub's API quota; respond to a
  demonstrated rate-limit failure with an optional token from runtime secrets
  or environment, or use raw HTTPS for the affected known files. Never hardcode
  a token.
- A named HTTP filesystem with `raw.githubusercontent.com` URLs avoids GitHub's
  repository API when only known public files are required. A bare
  `HTTPFileSystem` can open concrete URLs but has no listable root, so do not
  select it solely to expose a browsable source in marimo's Remote Storage
  panel.
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

For a conversion or runtime repair, then open the notebook in a fresh molab
session or a local sandbox. Confirm
that intended controls and embeds render, change each safe control at least
once, and inspect cell errors. Static checking cannot detect a nonexistent
runtime attribute such as `.clicked` or a widget that was constructed but
never returned for display. Keep remote-write controls unsubmitted, or use an
offline/test backend, during this smoke test.

For a narrow prose, link, or marker edit, verify the affected presentation and
targets and confirm executable code is unchanged. Do not rerun training solely
for that edit; report unrelated existing diagnostics separately.
