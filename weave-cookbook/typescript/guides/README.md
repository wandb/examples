# Guides — TypeScript

Ordered walkthroughs, one new idea each, run from the `typescript/` directory.

## Core path

| # | Guide | One new idea |
| --- | --- | --- |
| 00 | [Setup](00-setup/README.md) | `weave.init` + one conversation span logs your first exchange |
| 01 | [Tracing](01-tracing/README.md) | A turn is an `invoke_agent` trace with tool and LLM child spans |
| 02 | [Evaluations](02-evaluations/README.md) | Evaluation = named Dataset + Scorers + your app |
| 03 | [LLM as a judge](03-llm-judge/README.md) | A judge is a narrow, structured, calibrated scorer — never ground truth |

## Track agents

| # | Guide | One new idea |
| --- | --- | --- |
| 07 | [Track agents](07-agents/README.md) | Harness plugins trace agents you already use — Claude Code first, no code changes |

## Iterate

| # | Guide | One new idea |
| --- | --- | --- |
| 04 | [Version objects](04-versioning/README.md) | `client.publish` + `client.get` give any object an immutable history |
| 05 | [Prompts](05-prompts/README.md) | `StringPrompt` / `MessagesPrompt` with `{placeholders}`, published and versioned |

## Platform

| # | Guide | One new idea |
| --- | --- | --- |
| 06 | [Serverless Inference](06-serverless-inference/README.md) | One key, ~30 models, response settings, limits, and the Playground |
| 08 | [OpenTelemetry](08-otel/README.md) | Send spans from the plain OTel SDK to Weave's OTLP endpoint — no `weave` import |

The roadmap and phase status live in the [blueprint](../../reference/COOKBOOK_BLUEPRINT.md). Failing runs: [troubleshooting](../../reference/TROUBLESHOOTING.md).
