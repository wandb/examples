# Golden path

The shortest honest route through the cookbook, and the mental model it builds.

## Core path

1. **Initialize once.** `weave.init("<entity>/weave-cookbook")` connects the process to a project (guide 00). Everything traced afterward lands there.
2. **Trace conversations, not functions.** A conversation contains turns; a turn contains tool spans and LLM spans. That structure — not a pile of logs — is what the Agents view renders (guides 00–01).
3. **Measure before you believe.** An Evaluation runs your app over a dataset and records one score per quality dimension, deterministic checks first (guide 02). `EvaluationLogger` retrofits the same records onto loops you already have.
4. **Add a judge only for what code can't measure.** Narrow rubric, structured output, calibrated against human labels — and never treated as ground truth (guide 03).

## Track agents

Harness plugins log Claude Code and other agent sessions with no application-code changes (guide 07). This is a second on-ramp: track an agent you already use, not the next step in the Game Night loop.

## Iterate

Publish catalogs, datasets, and prompts when you need an immutable history; guide 05 evaluates two prompt versions against the same named dataset (guides 04–05).

## Optional branches

- **Inspect the model service.** List models, tune response settings, and open calls in Playground (guide 06).
- **Connect an existing pipeline.** Any OpenTelemetry SDK can send spans to Weave's OTLP endpoint without a `weave` dependency (guide 08).

Planned next: feedback capture, monitors, and production debugging. Status labels are kept current in the [blueprint](COOKBOOK_BLUEPRINT.md).
