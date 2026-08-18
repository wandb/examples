# OpenTelemetry

This optional, advanced integration sends live spans and requires `WANDB_ENTITY`. It uses the plain [OpenTelemetry SDK](https://opentelemetry.io/) — the path for services that already have an OTel pipeline, or that can't take a `weave` dependency.

Weave accepts OTLP traffic on a dedicated endpoint:

| | |
| --- | --- |
| URL | `https://trace.wandb.ai/otel/v1/traces` (`POST`, protobuf) |
| Auth | `wandb-api-key` header |
| Routing | `wandb.entity` and `wandb.project` resource attributes on the `TracerProvider` |

On Dedicated Cloud or self-managed instances the base URL is `https://<your-instance>.wandb.io/traces` instead.

## How it works

[`examples/08_otel.py`](../../examples/08_otel.py) builds an `OTLPSpanExporter` pointed at the endpoint and attaches it to a `TracerProvider` through a `BatchSpanProcessor` (the recommended processor — `SimpleSpanProcessor` exports synchronously and drags on your workload). `weave` is never imported:

```python
    exporter = OTLPSpanExporter(
        endpoint=OTLP_ENDPOINT,
        headers={"wandb-api-key": require_api_key()},
    )
    provider = TracerProvider(
        resource=Resource(
            {
                "wandb.entity": entity,
                "wandb.project": os.getenv("WANDB_PROJECT", DEFAULT_PROJECT),
            }
        )
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
```

The catalog search then runs inside two ordinary OTel spans, `recommend_game` and `search_catalog`. `provider.shutdown()` at the end matters: the batch processor flushes asynchronously, and a process that exits first drops spans without an error.

> **The endpoint pitfall.** The example passes the full path to `endpoint=` on the exporter, which uses it verbatim. If you configure the standard `OTEL_EXPORTER_OTLP_ENDPOINT` environment variable instead, the SDK appends `/v1/traces` on its own — so that variable must hold the base `https://trace.wandb.ai/otel`, or the request goes to a doubled path, gets a 404, and the spans are silently dropped.

## Run it

This guide needs `WANDB_ENTITY` uncommented and set in `.env`: with no Weave SDK in the process, nothing infers your default entity.

```bash
uv run python examples/08_otel.py
```

## What you should see

```text
Play Orchard Sprint: done in 20 minutes.
```

No weave banner and no trace link — the Weave SDK is not involved. The pick is deterministic (the quickest fitting game).

## Inspect it in Weave

Open **Traces** in your project. The newest trace is `recommend_game` with `search_catalog` nested inside — sent by pure OTel, sitting in the same list as SDK traces. Select a span: the `input.value` and `output.value` attributes appear in the side panel. Spans carrying `gen_ai.*` or `openinference.*` attributes get convention-aware rendering; everything else is shown as-is.

## Boundaries

- This endpoint feeds the **Traces** and **Threads** views. Spans for the **Agents** view go to a separate agents endpoint — see [Send OpenTelemetry spans to the Agents view](https://docs.wandb.ai/weave/guides/tracking/trace-agents-otel).
- In production, an [OpenTelemetry Collector](https://docs.wandb.ai/weave/guides/tracking/otel#use-an-opentelemetry-collector) can sit between your app and Weave to centralize auth and fan traces out to multiple backends.

Full capture: [python-08-otel.txt](../../../assets/expected-output/python-08-otel.txt).
