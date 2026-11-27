# Cookbook blueprint

The plan for the whole cookbook: what exists, what comes next, and the rules every addition follows. Contribution rules are in [CONTRIBUTING.md](../CONTRIBUTING.md).

## Canonical application

One small app carries the entire learning path: the **Game Night Agent**. It recommends a board game from a local JSON catalog (`examples/shared/data/games.json` in each language) based on player count, time budget, and the group's vibe. The domain is small enough to read in a minute but rich enough to support tool calls, a multi-turn follow-up, deterministic constraint checks, and an LLM judge later on.

## Phases

### Phase 1 — Tracing (built)

One trustworthy path from clone to a conversation in the Agents view.

| Guide | Idea | Status |
| --- | --- | --- |
| 00 — Setup | `weave.init` + one conversation span logs your first exchange | Built (Python, TypeScript) |
| 01 — Tracing | A turn is an `invoke_agent` trace with tool and LLM child spans | Built (Python, TypeScript) |

### Phase 2 — Evaluation (built)

| Guide | Idea | Status |
| --- | --- | --- |
| 02 — Evaluations | Evaluation = named Dataset + Scorers + your app | Built (Python, TypeScript) |
| 03 — LLM as a judge | A judge is a narrow, structured, calibrated scorer — never ground truth | Built (Python, TypeScript) |

The dataset keeps deliberately difficult rows (an impossible scenario whose right answer is "none", a batch row that fails `fits_constraints`). Do not tune them away. An optional `EvaluationLogger` recipe beside guide 02 covers existing evaluation loops without adding a second primary outcome to the guide.

### Track agents (built)

| Guide | Idea | Status |
| --- | --- | --- |
| 07 — Track agents | Harness plugins (Claude Code first) trace agents you already use, no code changes | Built (guide only; install is machine-level and deliberate) |

### Phase 3 — Iterate (built)

| Guide | Idea | Status |
| --- | --- | --- |
| 04 — Version objects | Publish and retrieve objects with an immutable version history | Built (Python, TypeScript) |
| 05 — Prompts | Version prompts, then compare two versions on the same named dataset | Built (Python, TypeScript) |

### Platform (built)

| Guide | Idea | Status |
| --- | --- | --- |
| 06 — Serverless Inference | One key, many models, response settings, limits, Playground | Built (Python, TypeScript) |
| 08 — OpenTelemetry | The plain OTel SDK sends spans to Weave's OTLP endpoint | Built (Python, TypeScript) |

### Phase 5 — Production signals (planned)

Feedback capture, monitors, and debugging tool errors and latency on live traffic.

## Rules every guide follows

- One guide, one new idea, one primary outcome.
- Complete examples that run as written from the language directory — no pseudocode, no `...` placeholders.
- Every guide ends with **Inspect it in Weave** and a concrete success state.
- Deterministic logic gets local tests; live behavior gets opt-in live tests.
- Boundaries are labeled honestly: Optional, Advanced, Experimental, Requires a live model.
- APIs are verified against official W&B sources before they appear here — see the [source map](SOURCE_MAP.md).
