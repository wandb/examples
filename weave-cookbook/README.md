# Weave Cookbook

Short, runnable Python and TypeScript guides for AI engineers learning [W&B Weave](https://weave-docs.wandb.ai/). Each guide adds one capability to the **Game Night Agent** and ends with a concrete result to inspect in Weave.

Official docs remain the API reference. This repository is the ordered path between a quickstart and a full production demo: enough code to trace, evaluate, and improve one LLM application, without deployment infrastructure or product boilerplate.

Published by [Lorenzo Balderrama](https://github.com/LorenzoWandB).

## Your first trace

Pick a language. Each cookbook is self-contained.

**Python** (needs [uv](https://docs.astral.sh/uv/)):

```bash
cd python
uv sync
cp .env.example .env   # set WANDB_API_KEY from https://wandb.ai/settings
uv run python examples/00_hello_trace.py
```

**TypeScript** (needs Node.js 20.12+):

```bash
cd typescript
npm install
cp .env.example .env   # set WANDB_API_KEY from https://wandb.ai/settings
npm run hello
```

Either way, the terminal prints a link to your project — open **Agents** to see the first conversation.

## Core path

| # | Guide | Python | TypeScript |
| --- | --- | --- | --- |
| 00 | Setup | [guide](python/guides/00-setup/README.md) | [guide](typescript/guides/00-setup/README.md) |
| 01 | Tracing | [guide](python/guides/01-tracing/README.md) | [guide](typescript/guides/01-tracing/README.md) |
| 02 | Evaluations | [guide](python/guides/02-evaluations/README.md) | [guide](typescript/guides/02-evaluations/README.md) |
| 03 | LLM as a judge | [guide](python/guides/03-llm-judge/README.md) | [guide](typescript/guides/03-llm-judge/README.md) |

The core loop is: trace the agent, measure it, add a judge. Version objects when you need to compare the next run.

## Track agents

| # | Guide | Python | TypeScript |
| --- | --- | --- | --- |
| 07 | Track agents | [guide](python/guides/07-agents/README.md) | [guide](typescript/guides/07-agents/README.md) |

Harness plugins trace agents you already use — Claude Code first, no code changes.

## Iterate

| # | Guide | Python | TypeScript |
| --- | --- | --- | --- |
| 04 | Version objects | [guide](python/guides/04-versioning/README.md) | [guide](typescript/guides/04-versioning/README.md) |
| 05 | Prompts | [guide](python/guides/05-prompts/README.md) | [guide](typescript/guides/05-prompts/README.md) |

## Platform

| # | Guide | Python | TypeScript |
| --- | --- | --- | --- |
| 06 | Serverless Inference | [guide](python/guides/06-serverless-inference/README.md) | [guide](typescript/guides/06-serverless-inference/README.md) |
| 08 | OpenTelemetry | [guide](python/guides/08-otel/README.md) | [guide](typescript/guides/08-otel/README.md) |

## I want to...

| Task | Where |
| --- | --- |
| Log my first conversation | Guide 00 ([Python](python/guides/00-setup/README.md) / [TypeScript](typescript/guides/00-setup/README.md)) |
| Trace tool and LLM calls inside a turn | Guide 01 ([Python](python/guides/01-tracing/README.md) / [TypeScript](typescript/guides/01-tracing/README.md)) |
| Track agents I already use | Guide 07 ([Python](python/guides/07-agents/README.md) / [TypeScript](typescript/guides/07-agents/README.md)) |
| Measure whether my app works | Guide 02 ([Python](python/guides/02-evaluations/README.md) / [TypeScript](typescript/guides/02-evaluations/README.md)) |
| Score a fuzzy property with an LLM judge | Guide 03 ([Python](python/guides/03-llm-judge/README.md) / [TypeScript](typescript/guides/03-llm-judge/README.md)) |
| Compare a new prompt or dataset version | Guides 04–05 ([Python](python/guides/04-versioning/README.md) / [TypeScript](typescript/guides/04-versioning/README.md)) |
| List or switch Serverless Inference models | Guide 06 ([Python](python/guides/06-serverless-inference/README.md) / [TypeScript](typescript/guides/06-serverless-inference/README.md)) |
| Send spans from an existing OTel pipeline | Guide 08 ([Python](python/guides/08-otel/README.md) / [TypeScript](typescript/guides/08-otel/README.md)) |
| Follow the shortest learning route | [Golden path](reference/GOLDEN_PATH.md) |
| See what is built and planned | [Cookbook blueprint](reference/COOKBOOK_BLUEPRINT.md) |
| Check tested SDK and runtime versions | [Compatibility](reference/COMPATIBILITY.md) |
| Understand Conversation, Turn, Scorer... | [Glossary](reference/GLOSSARY.md) |
| See where every API claim was verified | [Source map](reference/SOURCE_MAP.md) |
| Fix a failing run | [Troubleshooting](reference/TROUBLESHOOTING.md) |

## Map

```text
weave-cookbook/
├── python/       Python cookbook: examples, guides, tests
├── typescript/   TypeScript cookbook: examples, guides, tests
├── reference/    blueprint, glossary, source map, troubleshooting, compatibility
├── scripts/      documentation and secret checks
└── assets/       sanitized expected output
```

## Validate the repository

```bash
python3 scripts/validate_docs.py
python3 scripts/check_secrets.py
```

Per-language checks are listed in [python/README.md](python/README.md) and [typescript/README.md](typescript/README.md).

Contribution requirements are in [CONTRIBUTING.md](CONTRIBUTING.md). The examples are available under the [MIT License](LICENSE).
