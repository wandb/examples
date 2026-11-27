"""Send spans to Weave's OTLP endpoint with the OpenTelemetry SDK alone -- no weave import."""

import json
import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from shared.catalog import filter_games, load_catalog
from shared.config import DEFAULT_PROJECT, MissingConfigError, require_api_key

# The full path, because it is passed straight to the exporter. An
# OTEL_EXPORTER_OTLP_ENDPOINT environment variable would need the base
# URL instead: the SDK appends /v1/traces to it.
OTLP_ENDPOINT = "https://trace.wandb.ai/otel/v1/traces"


def provider_for_weave() -> TracerProvider:
    entity = os.getenv("WANDB_ENTITY")
    if not entity:
        raise MissingConfigError(
            "WANDB_ENTITY is not set. The OTLP endpoint routes spans by resource "
            "attributes, so this example needs an explicit entity in .env."
        )
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
    return provider


def recommend_game(tracer: trace.Tracer, players: int, minutes: int) -> str:
    with tracer.start_as_current_span("recommend_game") as span:
        span.set_attribute("input.value", json.dumps({"players": players, "minutes": minutes}))

        with tracer.start_as_current_span("search_catalog") as search:
            candidates = filter_games(load_catalog(), players=players, max_minutes=minutes)
            search.set_attribute("output.value", json.dumps([g["name"] for g in candidates]))

        if not candidates:
            reply = "No catalog game fits that group."
        else:
            quickest = min(candidates, key=lambda game: game["playtime_minutes"])
            reply = f"Play {quickest['name']}: done in {quickest['playtime_minutes']} minutes."

        span.set_attribute("output.value", json.dumps({"reply": reply}))
        return reply


def main() -> None:
    provider = provider_for_weave()
    tracer = provider.get_tracer("game-night-agent")
    print(recommend_game(tracer, players=4, minutes=60))
    provider.shutdown()  # flush the batch processor before the process exits


if __name__ == "__main__":
    main()
