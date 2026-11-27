# Contributing

Keep additions small, runnable, and aligned across Python and TypeScript.

Before writing a guide, read the [cookbook blueprint](reference/COOKBOOK_BLUEPRINT.md), [guide template](reference/GUIDE_TEMPLATE.md), and [source map](reference/SOURCE_MAP.md). A guide should add one idea, run from its language directory, show a concrete Weave success state, and link exhaustive API details to the official docs.

## Validate changes

From the repository root:

```bash
python3 scripts/validate_docs.py
python3 scripts/check_secrets.py
```

For Python:

```bash
cd python
uv sync
uv run ruff format --check .
uv run ruff check .
uv run pytest -q -m "not live"
```

For TypeScript:

```bash
cd typescript
npm ci
npm run typecheck
npm test
```

Live checks are opt-in because they write to W&B services and may use Inference credits. When an API, SDK behavior, UI label, or expected output changes, update the example, guide, tests, sanitized capture, [source map](reference/SOURCE_MAP.md), and [compatibility record](reference/COMPATIBILITY.md) together.
