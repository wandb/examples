# Compatibility

The environment every example was last verified against. Exact pins live in `python/uv.lock` and `typescript/package-lock.json`.

Last verified: 2026-08-02 on macOS (arm64).

## Python cookbook

| Component | Tested version | Minimum |
| --- | --- | --- |
| Python | 3.11.13 | 3.11 |
| weave | 0.53.4 | — |
| openai | 2.52.0 | — |
| opentelemetry-sdk / -exporter-otlp-proto-http | 1.44.0 | — |
| python-dotenv | 1.2.2 | — |
| pytest | 9.1.1 | — |
| ruff | 0.16.1 | — |

## TypeScript cookbook

| Component | Tested version | Minimum |
| --- | --- | --- |
| Node.js | 26.4.0 | 20.12 |
| weave (npm) | 0.16.5 | — |
| openai (npm) | 7.3.0 | — |
| @opentelemetry/sdk-trace-node, -base, resources | 1.30.1 | — |
| @opentelemetry/exporter-trace-otlp-proto | 0.53.0 | — |
| typescript | 7.0.2 | — |
| tsx | 4.23.4 | — |
| vitest | 4.1.10 | — |

The OTel packages are pinned to the release line `weave` (npm) already depends on, so only one copy of each ends up installed.

## Service endpoints

- W&B Serverless Inference: `https://api.inference.wandb.ai/v1`, default model `meta-llama/Llama-3.1-8B-Instruct`.
- Trace UI: `https://wandb.ai/<entity>/<project>/weave`.
- OTLP ingest (guide 08): `https://trace.wandb.ai/otel/v1/traces`, protobuf only.
- Claude Code plugin (guide 07): `weave-claude-code` 0.2.13 on npm at capture time; not installed by this repository.

## Known notes

- Live verification status of the examples is recorded in [SOURCE_MAP.md](SOURCE_MAP.md).
- When an SDK release changes an API or a UI label used here, update the affected example, guide, test, source map row, and this file together.
