import { trace } from '@opentelemetry/api';
import { describe, expect, it } from 'vitest';

import {
  recommendGame as recommendWithAgentSpans,
  runConversation,
} from '../examples/01-trace-recommendation.js';
import { CALIBRATION } from '../examples/03-llm-judge.js';
import {
  requestPrompt,
  systemPrompt,
  vibeFirstSystemPrompt,
} from '../examples/05-prompts.js';
import { recommendGame as recommendWithOTel } from '../examples/08-otel.js';
import { loadCatalog } from '../examples/shared/catalog.js';
import {
  fitsConstraints,
  type Scenario,
  validPick,
} from '../examples/shared/evaluation.js';

const cooperativeScenario: Scenario = {
  players: 4,
  minutes: 60,
  vibe: 'cooperative',
};

describe('cookbook examples', () => {
  it('returns none before making a model call when no game fits', async () => {
    await expect(recommendWithAgentSpans(4, 10, 'quick')).resolves.toBe('none');
  });

  it('keeps a follow-up under one conversation', async () => {
    await expect(runConversation(4, 10, 'quick', 5)).resolves.toEqual(['none', 'none']);
  });

  it('scores valid and invalid recommendations deterministically', async () => {
    await expect(
      validPick({ datasetRow: cooperativeScenario, modelOutput: 'Signal Lost' }),
    ).resolves.toEqual({ valid_pick: true });
    await expect(
      fitsConstraints({ datasetRow: cooperativeScenario, modelOutput: 'Ironroot' }),
    ).resolves.toEqual({ fits_constraints: false });
  });

  it('formats every prompt placeholder', () => {
    const system = systemPrompt.format({ vibe: 'cooperative' });
    const vibeFirst = vibeFirstSystemPrompt.format({ vibe: 'cooperative' });
    const messages = requestPrompt.format({
      players: 4,
      minutes: 60,
      candidates: '- Signal Lost',
    });

    expect(system).toContain('cooperative');
    expect(system).not.toContain('{vibe}');
    expect(vibeFirst).toContain('cooperative');
    expect(vibeFirst).not.toContain('{vibe}');
    expect(messages[0].content).toContain('Players: 4');
    expect(messages[0].content).toContain('- Signal Lost');
  });

  it('keeps judge calibration rows labeled and in the catalog', () => {
    const names = new Set(loadCatalog().map((game) => game.name));

    expect(CALIBRATION.every((row) => names.has(row.game))).toBe(true);
    expect(new Set(CALIBRATION.map((row) => row.human_says))).toEqual(new Set([true, false]));
  });

  it('runs the deterministic OTel path without a provider', () => {
    const tracer = trace.getTracer('test');

    expect(recommendWithOTel(tracer, 4, 60)).toBe('Play Orchard Sprint: done in 20 minutes.');
    expect(recommendWithOTel(tracer, 4, 10)).toContain('No catalog game fits');
  });
});
