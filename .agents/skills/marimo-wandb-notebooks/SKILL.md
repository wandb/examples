---
name: marimo-wandb-notebooks
description: Convert existing Jupyter or Colab .ipynb tutorials in wandb/examples into repo-ready marimo .py notebooks suitable for molab. Use when converting notebooks, reviewing or cleaning up pre-converted marimo notebooks, diagnosing saved conversion logs, fixing marimo check failures, or refactoring Jupyter execution-order dependencies for marimo while preserving the original tutorial's W&B teaching value.
---

# Convert W&B example notebooks to marimo

## Determine the starting state

### `.ipynb`

If no usable marimo conversion exists:

1. Inspect the source notebook.
2. From the repository root, run:

  ```bash
  scripts/convert-colab-to-marimo.py <notebook.ipynb> --name <example-name>
  ```

3. Inspect the generated `.py` and `.logs/`.

### Pre-converted `.py`

Resume from the existing conversion:

1. Inspect the `.py` and nearby `.logs/`.
2. Read `.logs/result.json` first when available, then the relevant stage log.
3. Run a fresh `uvx marimo check <notebook.py>`.

Do not reconvert solely because the original `.ipynb` exists. Treat saved
logs as diagnostic history, not current state. Reconvert only when explicitly
requested or when the existing conversion is unusable.

## Reference routing

Before modifying a notebook, read:

- [`references/marimo-idioms.md`](references/marimo-idioms.md)
- [`references/tutorial-notebook-objectives.md`](references/tutorial-notebook-objectives.md)

Also read:

- [`references/convert-cleanup.md`](references/convert-cleanup.md) for
  generated, pre-converted, or failed conversions.
- [`references/wandb-patterns.md`](references/wandb-patterns.md) when
  reviewing or changing W&B SDK usage.

## Sources of truth

- Use the original `.ipynb` for tutorial intent, narrative, and featured W&B
  behavior.
- Treat the current `.py` as the current conversion state.
- Treat `.logs/` as historical diagnostic evidence.
- Use a fresh `marimo check` for current static validity.
- Use `examples/marimo/mnist-registry/mnist_registry.py` as the structural
  exemplar when the references do not specify a choice, especially its
  separation of marimo orchestration cells from reusable `@app.function`
  helpers. Do not copy tutorial-specific details from the exemplar.

## Conversion priorities

1. Produce valid marimo with a correct reactive graph.
2. Preserve tutorial behavior and a clean teaching surface; keep featured W&B
   SDK usage inspectable.
3. Remove conversion artifacts and unnecessary global state.
4. Apply repository conventions and polish the reader experience.

Fix `marimo check` blockers first. Passing the check is necessary, but does
not by itself complete the conversion.

## Repo conventions

- Keep each completed example in `examples/marimo/<example-name>/` with the
  notebook named `<example_name>.py`.
- Treat the notebook `.py` as authoritative; never edit its generated `.md`
  export.
- Do not commit runtime-generated files such as `data/`, `wandb/`,
  `artifacts/`, `__marimo__/`, model weights, or similar outputs.
- Use `.logs/` only for diagnosis; final notebooks and docs must not depend on
  them.

## Review-only tasks

When reviewing without editing, inspect the notebook and relevant logs, run a
fresh `marimo check` when possible, and compare with the source `.ipynb` when
needed.

Report material issues in this order:

1. correctness and marimo blockers;
2. tutorial fidelity;
3. W&B SDK usage;
4. repository conventions.

Give a concrete fix for each issue. Do not modify files unless asked.

## Final verification

Before considering a conversion complete:

- `uvx marimo check <notebook.py>` passes after the final notebook edit; do
  not rely on a saved conversion log or an earlier successful check.
- A fresh sandboxed local session or molab session opens without cell errors.
  Every intended widget, form, and embedded player visibly renders, and each
  orchestration cell reads a documented reactive value. Keep remote-write
  controls unsubmitted, or use an offline/test backend, during this smoke test.
- Notebook structure follows
  [`references/marimo-idioms.md`](references/marimo-idioms.md), including the
  separation of teaching code, marimo orchestration, and reusable helpers.
- Tutorial quality follows
  [`references/tutorial-notebook-objectives.md`](references/tutorial-notebook-objectives.md).
- Featured W&B SDK usage follows
  [`references/wandb-patterns.md`](references/wandb-patterns.md).
- No unintended generated or runtime files were introduced, and the notebook
  and docs do not depend on `.logs/`.
- Opening the notebook and changing unsubmitted controls do not create W&B
  objects or other remote side effects.
- Each submission performs its intended remote writes once; re-submission
  closes prior runs and avoids duplicate stateful updates where practical.
- Cross-cell consumers of W&B state use explicit result or completion values,
  and producers synchronize server-visible state before downstream reads.
- A fresh molab session can authenticate without credentials from the local
  computer, and no credential appears in notebook output, run configuration,
  logs, or the diff.
