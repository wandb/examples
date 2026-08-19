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
4. Read [`references/convert-cleanup.md`](references/convert-cleanup.md).

### Pre-converted `.py`

Resume from the existing conversion:

1. Inspect the `.py` and nearby `.logs/`.
2. Read `.logs/result.json` first when available, then the relevant stage log.
3. Run a fresh `uvx marimo check <notebook.py>`.
4. Read [`references/convert-cleanup.md`](references/convert-cleanup.md).

Do not reconvert solely because the original `.ipynb` exists. Treat saved
logs as diagnostic history, not current state. Reconvert only when explicitly
requested or when the existing conversion is unusable.

## Reference routing

Before modifying a notebook, read:

* [`references/marimo-idioms.md`](references/marimo-idioms.md)
* [`references/tutorial-notebook-objectives.md`](references/tutorial-notebook-objectives.md)

Also read:

* [`references/convert-cleanup.md`](references/convert-cleanup.md) for
  generated, pre-converted, or failed conversions.
* [`references/wandb-patterns.md`](references/wandb-patterns.md) when
  reviewing or changing W&B API usage.

## Sources of truth

* Original `.ipynb`: tutorial intent, narrative, and featured W&B behavior.
* Current `.py`: current conversion state.
* `.logs/`: historical diagnostic evidence.
* Fresh `marimo check`: current static validity.
* `examples/marimo/mnist-registry/mnist_registry.py`: structural exemplar
  when the references do not specify a choice. Do not copy tutorial-specific
  details from the exemplar.

## Conversion priorities

1. Produce valid marimo with a correct reactive graph.
2. Preserve tutorial behavior, teaching value, and visible featured W&B APIs.
3. Remove marimo conversion artifacts and unnecessary global state.
4. Apply repository conventions and polish the reader experience.

Fix `marimo check` blockers first. Passing the check is necessary, but does
not by itself complete the conversion.

## Repo conventions

* Keep each completed example in `examples/marimo/<example-name>/` with the
  notebook named `<example_name>.py`.
* Treat `.py` as authoritative; never edit generated `.md` exports.
* Do not commit runtime-generated files such as `data/`, `wandb/`,
  `artifacts/`, `__marimo__/`, model weights, or similar outputs.
* Use `.logs/` only for diagnosis; final notebooks and docs must not depend
  on them.

## Review-only tasks

When reviewing without editing, inspect the notebook and relevant logs, run a
fresh `marimo check` when possible, and compare with the source `.ipynb` when
needed.

Report material issues in this order:

1. correctness and marimo blockers;
2. tutorial fidelity;
3. W&B API usage;
4. repository conventions.

Give a concrete fix for each issue. Do not modify files unless asked.

## Final verification

Before considering a conversion complete:

* `uvx marimo check <notebook.py>` passes.
* The globals audit in
  [`references/marimo-idioms.md`](references/marimo-idioms.md) passes.
* The tutorial-objectives audit in
  [`references/tutorial-notebook-objectives.md`](references/tutorial-notebook-objectives.md)
  passes.
* Featured W&B APIs follow
  [`references/wandb-patterns.md`](references/wandb-patterns.md).
* No unintended generated or runtime files were introduced.
* The notebook and docs do not depend on `.logs/`.
