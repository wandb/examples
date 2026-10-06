// Parameterized, versioned prompts: the agent's wording becomes a tracked object.

import { parseArgs } from 'node:util';
import OpenAI from 'openai';
import * as weave from 'weave';
import { StringPrompt } from 'weave';

import { describeGame, filterGames, loadCatalog } from './shared/catalog.js';
import { inferenceClient, loadEnv, modelId, weaveProject } from './shared/config.js';
import {
  fitsConstraints,
  SCENARIOS,
  type Scenario,
  validPick,
} from './shared/evaluation.js';
import { isDirectExecution } from './shared/runtime.js';

export const systemPrompt = new weave.StringPrompt({
  content:
    'You are the Game Night Agent. Pick exactly one game from the candidate list. ' +
    'Choose the best match for a group that wants {vibe}. ' +
    'Reply with the game name only, exactly as written in the list.',
});

export const requestPrompt = new weave.MessagesPrompt({
  messages: [
    {
      role: 'user',
      content:
        'Players: {players}\nTime available: {minutes} minutes\n\nCandidates:\n{candidates}',
    },
  ],
});

export const vibeFirstSystemPrompt = new weave.StringPrompt({
  content:
    'You are the Game Night Agent. Pick exactly one game from the candidate list. ' +
    'Prioritize the requested vibe ({vibe}), then difficulty. ' +
    'Reply with the game name only, exactly as written in the list.',
});

async function recommendWithPrompt(prompt: weave.StringPrompt, scenario: Scenario): Promise<string> {
  const candidates = filterGames(loadCatalog(), {
    players: scenario.players,
    maxMinutes: scenario.minutes,
  });
  if (candidates.length === 0) {
    return 'none';
  }
  const messages = [
    { role: 'system', content: prompt.format({ vibe: scenario.vibe }) },
    ...requestPrompt.format({
      players: scenario.players,
      minutes: scenario.minutes,
      candidates: candidates.map(describeGame).join('\n'),
    }),
  ] as OpenAI.Chat.Completions.ChatCompletionMessageParam[];
  const response = await weave.wrapOpenAI(inferenceClient()).chat.completions.create({
    model: modelId(),
    messages,
  });
  return (response.choices[0].message.content ?? '').trim().replace(/^"|"$/g, '');
}

async function comparePrompts(client: Awaited<ReturnType<typeof weave.init>>): Promise<void> {
  await client.publish(vibeFirstSystemPrompt, 'game-night-system');
  const dataset = new weave.Dataset({ name: 'game-night-scenarios', rows: SCENARIOS });
  const prompts = [
    ['baseline', systemPrompt],
    ['vibe-first', vibeFirstSystemPrompt],
  ] as const;

  for (const [label, prompt] of prompts) {
    const model = weave.op(
      ({ datasetRow }: { datasetRow: Scenario }) => recommendWithPrompt(prompt, datasetRow),
      { name: `GameNightModel_${label}` },
    );
    const evaluation = new weave.Evaluation({
      name: `game-night-prompt-${label}`,
      dataset,
      scorers: [validPick, fitsConstraints],
    });
    const summary = await evaluation.evaluate({ model });
    const valid = summary.valid_pick.valid_pick.true_fraction;
    const fits = summary.fits_constraints.fits_constraints.true_fraction;
    console.log(`${label}: valid_pick=${Math.round(valid * 100)}%, fits_constraints=${Math.round(fits * 100)}%`);
  }
}

export async function main(): Promise<void> {
  loadEnv();
  const { values } = parseArgs({
    options: {
      compare: { type: 'boolean', default: false },
    },
  });
  const client = await weave.init(weaveProject());

  const systemRef = await client.publish(systemPrompt, 'game-night-system');
  await client.publish(requestPrompt, 'game-night-request');

  if (values.compare) {
    await comparePrompts(client);
  } else {
    console.log(
      await recommendWithPrompt(systemPrompt, {
        players: 4,
        minutes: 60,
        vibe: 'cooperative',
      }),
    );

    // Application code can load wording by reference instead of hardcoding it.
    const fetched = await StringPrompt.get(client, systemRef.uri());
    console.log(
      `\nfetched game-night-system -> ${fetched.format({ vibe: 'competitive' }).slice(0, 80)}...`,
    );
  }
}

if (isDirectExecution(import.meta.url)) {
  await main();
}
