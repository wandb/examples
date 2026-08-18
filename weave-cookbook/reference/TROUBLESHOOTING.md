# Troubleshooting

Failures the examples are known to produce, and what fixes them.

## "WANDB_API_KEY is not set"

The examples read `.env` from the directory you run them in.

- Run commands from the language directory (`python/` or `typescript/`), not the repository root.
- Confirm `.env` exists there (`cp .env.example .env`) and the key is filled in from [wandb.ai/settings](https://wandb.ai/settings).

## 401 Unauthorized from api.inference.wandb.ai

The key is present but wrong or revoked. Generate a fresh key at [wandb.ai/settings](https://wandb.ai/settings) and update `.env`.

## "ValueError: entity_name must be non-empty" on weave.init

`WANDB_ENTITY` is present in your `.env` but empty, so the SDK reads an empty entity instead of falling back to your default team. Delete the line (or comment it out) unless you are filling in a real team name. `.env.example` ships with it commented for this reason.

## 403 or "permission denied" on weave.init

`WANDB_ENTITY` names a team you do not belong to (or is misspelled). Set it to a team you are a member of, or leave it empty to use your default team.

## Traces go to an unexpected place

The project path is printed at startup ("View Weave data at ..."). It is built from `WANDB_ENTITY` and `WANDB_PROJECT` (default `weave-cookbook`). Adjust those in `.env`.

## Model not found

`MODEL_ID` must be an ID from the [available models list](https://docs.wandb.ai/guides/inference/models/). The tested default is `meta-llama/Llama-3.1-8B-Instruct`.

## 429 Too Many Requests / quota exceeded

You have hit your W&B Inference usage limits. Check [usage information and limits](https://docs.wandb.ai/inference/usage-limits) for your plan and credits.

## `uv: command not found` / Node version errors

- Install uv from [docs.astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/); it provisions Python 3.11 automatically on `uv sync`.
- The TypeScript cookbook needs Node.js 20.12+ (`node --version`) because the examples use `process.loadEnvFile()`.

## A run works but no trace appears

Confirm the terminal printed a "View Weave data at ..." line (Python also prints a 🍩 link for op-based calls). If it did, you are looking at a different project or entity in the UI — open the exact printed URL, and remember conversations live under **Agents**, not **Traces**. If it did not, `weave.init` was not reached before the traced code; run the example file as written. In TypeScript, span-based examples must also end with `await weave.flushOTel()` — exiting first drops spans silently.

## OTel spans (guide 08) never show up

Three causes, in the order to check them:

- `WANDB_ENTITY` is unset. The OTLP endpoint routes by the `wandb.entity` and `wandb.project` resource attributes; the examples fail early with a message when the entity is missing.
- The endpoint has a doubled path. If you use the `OTEL_EXPORTER_OTLP_ENDPOINT` environment variable, it must hold `https://trace.wandb.ai/otel` — the SDK appends `/v1/traces` on its own. The full path belongs only in an explicit `endpoint=` / `url:` argument. A doubled path returns 404 and drops spans silently.
- The process exited before the flush. `BatchSpanProcessor` exports asynchronously; the examples call `provider.shutdown()` last for exactly this reason.
