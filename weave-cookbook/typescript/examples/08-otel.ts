// Send spans to Weave's OTLP endpoint with the OpenTelemetry SDK alone -- no weave import.

import type { Tracer } from '@opentelemetry/api';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-proto';
import { Resource } from '@opentelemetry/resources';
import { BatchSpanProcessor } from '@opentelemetry/sdk-trace-base';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';

import { filterGames, loadCatalog } from './shared/catalog.js';
import { DEFAULT_PROJECT, loadEnv, requireApiKey } from './shared/config.js';
import { isDirectExecution } from './shared/runtime.js';

// The full path, because it is passed straight to the exporter. An
// OTEL_EXPORTER_OTLP_ENDPOINT environment variable would need the base
// URL instead: the SDK appends /v1/traces to it.
const OTLP_ENDPOINT = 'https://trace.wandb.ai/otel/v1/traces';

function providerForWeave(): NodeTracerProvider {
  const entity = process.env.WANDB_ENTITY;
  if (!entity) {
    console.error(
      'WANDB_ENTITY is not set. The OTLP endpoint routes spans by resource ' +
        'attributes, so this example needs an explicit entity in .env.',
    );
    process.exit(1);
  }
  const exporter = new OTLPTraceExporter({
    url: OTLP_ENDPOINT,
    headers: { 'wandb-api-key': requireApiKey() },
  });
  const provider = new NodeTracerProvider({
    resource: new Resource({
      'wandb.entity': entity,
      'wandb.project': process.env.WANDB_PROJECT ?? DEFAULT_PROJECT,
    }),
  });
  provider.addSpanProcessor(new BatchSpanProcessor(exporter));
  provider.register();
  return provider;
}

export function recommendGame(tracer: Tracer, players: number, minutes: number): string {
  return tracer.startActiveSpan('recommend_game', (span) => {
    try {
      span.setAttribute('input.value', JSON.stringify({ players, minutes }));

      const candidates = tracer.startActiveSpan('search_catalog', (search) => {
        try {
          const found = filterGames(loadCatalog(), { players, maxMinutes: minutes });
          search.setAttribute('output.value', JSON.stringify(found.map((game) => game.name)));
          return found;
        } finally {
          search.end();
        }
      });

      let reply: string;
      if (candidates.length === 0) {
        reply = 'No catalog game fits that group.';
      } else {
        const quickest = candidates.reduce((best, game) =>
          game.playtime_minutes < best.playtime_minutes ? game : best,
        );
        reply = `Play ${quickest.name}: done in ${quickest.playtime_minutes} minutes.`;
      }

      span.setAttribute('output.value', JSON.stringify({ reply }));
      return reply;
    } finally {
      span.end();
    }
  });
}

export async function main(): Promise<void> {
  loadEnv();
  const provider = providerForWeave();
  console.log(recommendGame(provider.getTracer('game-night-agent'), 4, 60));
  await provider.shutdown(); // flush the batch processor before the process exits
}

if (isDirectExecution(import.meta.url)) {
  await main();
}
