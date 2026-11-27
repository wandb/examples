# Glossary

Quick definitions for the terms this cookbook uses. The exhaustive reference is the [official Weave documentation](https://weave-docs.wandb.ai/).

## Agent tracing (guides 00–01)

The semantic model that makes multi-turn behavior appear in Weave's Agents view:

- **Agent** — the top-level application identity, named when a conversation starts.
- **Conversation** — one session between a user and the agent. Multi-turn sessions are linked under one conversation ID.
- **Turn** — one user-visible exchange inside a Conversation. Each turn becomes an `invoke_agent` trace.
- **LLM span / Tool span** — children of a turn recording one model call (`chat`) or one tool call (`execute_tool`), with inputs, outputs, and token usage.

See [Trace your agents](https://docs.wandb.ai/weave/guides/tracking/trace-agents).

## Function tracing (guides 02–03)

- **Op** — a versioned, tracked function. Created with `@weave.op()` in Python or `weave.op(fn)` in TypeScript. Weave stores its source code and versions it when the code changes. The evaluation guides use Ops for models and scorers.
- **Call** — one logged execution of an Op: inputs, output, timing, errors, and its parent-child relationships.
- **Trace** — the full tree of Calls (or imported OTel spans) from one execution context.
- **Thread** — related traces grouped into one session.

See [Understand Ops, Calls, and Traces](https://docs.wandb.ai/weave/guides/tracking/tracing).

## Evaluation (guides 02–03)

**Evaluation = Dataset + one or more Scorers + the application being tested.**

- **Dataset** — reusable, versioned examples representing expected behavior, important scenarios, or known failures.
- **Scorer** — a function that measures one property of an output, such as `fits_constraints`. Deterministic first; an LLM judge only for properties code cannot reasonably measure.
- **Evaluation** — the reusable definition that runs an application over a Dataset and records scores.
- **Evaluation run** — one execution of an Evaluation against a particular application version.
- **EvaluationLogger** — the imperative alternative: log predictions and scores one at a time from an existing loop, without restructuring it into an Evaluation.
- **LLM judge** — a scorer that asks a model a narrow question with structured output. Calibrated against human labels; never ground truth.

See [Weave evaluations](https://docs.wandb.ai/weave/guides/core-types/evaluations).

## Objects and versions (guides 04–05)

- **Published object** — any value stored in Weave with `weave.publish` (Python) or `client.publish` (TypeScript). Re-publishing under the same name creates a new immutable version (`v0`, `v1`, ...).
- **Ref** — the URI that names an object version; `weave.ref("name:v0").get()` retrieves it, and bare names resolve to the latest version.
- **Prompt** — a published, parameterized wording object: `StringPrompt` for one string, `MessagesPrompt` for a message list, both with `{placeholder}` formatting.

## Interop (guides 07–08)

- **Agent harness** — an end-user agent runtime (Claude Code, Codex, OpenClaw, Pi). A Weave plugin traces its sessions without code changes.
- **OTLP endpoint** — `https://trace.wandb.ai/otel/v1/traces`, which accepts protobuf spans from any OpenTelemetry SDK; `wandb.entity` and `wandb.project` resource attributes route them.
