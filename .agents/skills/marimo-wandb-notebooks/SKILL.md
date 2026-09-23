---
name: marimo-wandb-notebooks
description: Convert existing Jupyter or Colab .ipynb tutorials in wandb/examples into repo-ready marimo .py notebooks suitable for molab. Use when converting notebooks, reviewing or cleaning up pre-converted marimo notebooks, diagnosing saved conversion logs, fixing marimo check failures, or refactoring Jupyter execution-order dependencies for marimo while preserving the original tutorial's W&B teaching value.
---

# Convert W&B example notebooks to marimo

## Why we are migrating

**Mission**: Make marimo *the* **standard** computational notebook, supporting
the entire lifecycle of computational science and development.

These migrations bring established W&B workflows into marimo: learning,
experimentation, training, debugging, inspecting results, and sharing
reproducible work. Preserve the tutorials' teaching value while making those
workflows usable end to end in marimo and molab. A successful migration gives
readers a notebook they can understand, run, adapt, and build on throughout
that lifecycle.

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
  generated, pre-converted, or failed conversions, molab links, and hosted
  runtime failures.
- [`references/wandb-patterns.md`](references/wandb-patterns.md) when
  reviewing or changing W&B SDK usage, including logged media.

## Sources of truth

- Use the original `.ipynb` for tutorial intent, narrative, and featured W&B
  behavior.
- Treat the current `.py` as the current conversion state.
- When the user designates a live marimo or molab notebook, or an artifact
  exported from it, as authoritative, it overrides the repository `.py` as the
  current implementation state. The original `.ipynb` remains a reference for
  intent, not a reason to rewrite the designated artifact. Work on the artifacts
  the user requested: a live repair does not imply a repository sync, while a
  request to fix both requires both. When syncing is requested,
  export or download after the final live edit and preserve cell order,
  boundaries, setup status, identifiers, dependency metadata, working
  integrations, and configuration such as `hide_code` and `disabled` unless the
  request requires changing them or a demonstrated runtime issue requires a
  dependency correction.
- Treat `.logs/` as historical diagnostic evidence.
- Use a fresh `marimo check` for current static validity.
- Use `examples/marimo/mnist-registry/mnist_registry.py` as the structural
  exemplar when the references do not specify a choice, especially its
  separation of marimo orchestration cells from reusable `@app.function`
  helpers. Do not copy tutorial-specific details from the exemplar.
- Notebook review or repair does not authorize submitting live credentials or
  remote-write controls, committing, pushing, or changing a pull request.
  Perform those actions only when the user explicitly requests them.

## Conversion priorities

1. Produce valid marimo with a correct reactive graph.
2. Preserve tutorial behavior and a clean teaching surface; keep featured W&B
   SDK usage inspectable.
3. Remove conversion artifacts and unnecessary global state.
4. Apply repository conventions and polish the reader experience.

Fix `marimo check` blockers first. Passing the check is necessary, but does
not by itself complete the conversion.

## Repo conventions

- Follow the parent branch's placement convention. In the conversion workflow,
  keep notebooks in `marimo/convert/<example-name>/<example_name>.py`; do not
  move them into `examples/marimo/` merely because cleanup is complete. Preserve
  other existing locations unless the user requests relocation.
- Do not add per-notebook READMEs, reports, or other repository artifacts unless
  requested. The notebook itself is the teaching surface.
- Treat the notebook `.py` as authoritative; never edit its generated `.md`
  export.
- Do not commit runtime-generated files such as `data/`, `wandb/`,
  `artifacts/`, `__marimo__/`, model weights, or similar outputs.
- Use `.logs/` only for diagnosis; final notebooks and docs must not depend on
  them.
- Preserve source comments, docstrings, lesson order, and model demonstrations;
  do not editorialize teaching code during conversion. Follow
  [`tutorial-notebook-objectives.md`](references/tutorial-notebook-objectives.md).
- Keep molab launch badges and useful cross-notebook links, targeting converted
  files. Remove obsolete Colab self-links and `@wandbcode` HTML markers using
  [`convert-cleanup.md`](references/convert-cleanup.md#notebook-links-and-html-markers).
- Treat a request covering all notebooks as a repository-wide marimo scan,
  not just the current batch. Do not edit the original `.ipynb` sources or
  generated session files unless they are explicitly in scope.

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

Scale verification to the change. For a narrow prose, link, or marker cleanup,
check every changed file, verify link targets, and confirm executable code is
unchanged; do not retrain models for a documentation-only edit. Distinguish
pre-existing diagnostics from new regressions. For conversions and runtime
repairs, validate the affected execution path as well as static validity.

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
- The setup cell remains visible while implementation-only authentication,
  status, and embed cells are hidden where their rendered output is the
  reader-facing surface.
- Tutorial quality follows
  [`references/tutorial-notebook-objectives.md`](references/tutorial-notebook-objectives.md).
- Compare every notebook in the requested scope with its source, including
  nested callbacks and small inline comments. Confirm that removed cells were
  obsolete or empty, not missing teaching steps.
- The Markdown outline has exactly one level-one heading for the notebook title;
  major sections use level two, and nested sections do not skip heading levels.
- Featured W&B SDK usage follows
  [`references/wandb-patterns.md`](references/wandb-patterns.md).
- No unintended generated or runtime files were introduced, and the notebook
  and docs do not depend on `.logs/`.
- Each converted notebook has one molab badge pointing to itself. Cross-links
  resolve to the intended converted notebook and the requested published ref;
  use `main` when explicitly preparing links for after merge.
- Opening the notebook and changing unsubmitted controls do not create W&B
  objects or other remote side effects.
- Each submission performs its intended remote writes once; re-submission
  closes prior runs and avoids duplicate stateful updates where practical.
- Cross-cell consumers of W&B state use explicit result or completion values,
  and producers synchronize server-visible state before downstream reads.
- A fresh molab session presents an authentication path that does not assume
  credentials from the local computer. Inspect the inputs and gate without
  submitting real credentials; test live authentication only when explicitly
  requested or with a designated test backend. No credential appears in
  notebook output, run configuration, logs, or the diff.
