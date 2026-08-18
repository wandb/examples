# Weave Cookbook — TypeScript

Runnable, progressive examples for [W&B Weave](https://weave-docs.wandb.ai/): trace, evaluate, and improve an LLM application. Everything is built around one small app, the Game Night Agent, which recommends a board game from a local catalog.

Published by [Lorenzo Balderrama](https://github.com/LorenzoWandB).

## First trace in three commands

From this directory (Node.js 20.12+):

```bash
npm install
cp .env.example .env   # then set WANDB_API_KEY from https://wandb.ai/settings
npm run hello
```

The terminal prints a link to your project — open **Agents** to see the conversation.

## Core path

| Guide | One new idea |
| --- | --- |
| [00 — Setup](guides/00-setup/README.md) | `weave.init` + one conversation span logs your first exchange |
| [01 — Tracing](guides/01-tracing/README.md) | A turn is an `invoke_agent` trace with tool and LLM child spans |
| [02 — Evaluations](guides/02-evaluations/README.md) | Evaluation = named Dataset + Scorers + your app |
| [03 — LLM as a judge](guides/03-llm-judge/README.md) | A judge is a narrow, structured, calibrated scorer — never ground truth |

## Track agents

| Guide | One new idea |
| --- | --- |
| [07 — Track agents](guides/07-agents/README.md) | Harness plugins trace agents you already use — Claude Code first, no code changes |

## Iterate

| Guide | One new idea |
| --- | --- |
| [04 — Version objects](guides/04-versioning/README.md) | `client.publish` + `client.get` give any object an immutable history |
| [05 — Prompts](guides/05-prompts/README.md) | `StringPrompt` / `MessagesPrompt` with `{placeholders}`, published and versioned |

## Platform

| Guide | One new idea |
| --- | --- |
| [06 — Serverless Inference](guides/06-serverless-inference/README.md) | One key, ~30 models, response settings, limits, and the Playground |
| [08 — OpenTelemetry](guides/08-otel/README.md) | Send spans from the plain OTel SDK to Weave's OTLP endpoint — no `weave` import |

## I want to...

| Task | Where |
| --- | --- |
| Log my first conversation | [Guide 00](guides/00-setup/README.md) |
| Trace tool and LLM calls inside a turn | [Guide 01](guides/01-tracing/README.md) |
| Track agents I already use | [Guide 07](guides/07-agents/README.md) |
| Measure whether my app works | [Guide 02](guides/02-evaluations/README.md) |
| Score a fuzzy property with an LLM judge | [Guide 03](guides/03-llm-judge/README.md) |
| Compare a new prompt or dataset version | [Guides 04–05](guides/04-versioning/README.md) |
| List or switch Serverless Inference models | [Guide 06](guides/06-serverless-inference/README.md) |
| Send spans from an existing OTel pipeline | [Guide 08](guides/08-otel/README.md) |
| Follow the shortest learning route | [Golden path](../reference/GOLDEN_PATH.md) |
| See what is built and planned | [Cookbook blueprint](../reference/COOKBOOK_BLUEPRINT.md) |
| Check tested SDK and runtime versions | [Compatibility](../reference/COMPATIBILITY.md) |
| Look up a term (Conversation, Turn, Scorer...) | [Glossary](../reference/GLOSSARY.md) |
| Fix a failing run | [Troubleshooting](../reference/TROUBLESHOOTING.md) |

## Map

```text
typescript/
├── examples/            complete runnable code
│   ├── 00-hello-trace.ts
│   ├── 01-trace-recommendation.ts
│   ├── 02-evaluate.ts / 02-eval-logger.ts
│   ├── 03-llm-judge.ts
│   ├── 04-versioning.ts
│   ├── 05-prompts.ts
│   ├── 06-inference.ts
│   ├── 08-otel.ts       (07 is a harness guide; it has no example file)
│   └── shared/          env config, catalog helpers, games.json
├── guides/              ordered walkthroughs
└── tests/               deterministic tests (vitest)
```

## Validate locally

```bash
npm run typecheck
npm test
```
