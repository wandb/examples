// Evaluate the Game Night Agent: a dataset, two deterministic scorers, one model function.

import * as weave from 'weave';

import { describeGame, filterGames, loadCatalog } from './shared/catalog.js';
import { inferenceClient, loadEnv, modelId, weaveProject } from './shared/config.js';
import { fitsConstraints, SCENARIOS, type Scenario, validPick } from './shared/evaluation.js';

const SYSTEM_PROMPT =
  'You are the Game Night Agent. Pick the best game for the group from the candidate list. ' +
  'Reply with the game name only, exactly as written in the list.';

const gameNightModel = weave.op(
  async ({ datasetRow }: { datasetRow: Scenario }) => {
    const candidates = filterGames(loadCatalog(), {
      players: datasetRow.players,
      maxMinutes: datasetRow.minutes,
    });
    if (candidates.length === 0) {
      return 'none';
    }
    const prompt =
      `Players: ${datasetRow.players}\n` +
      `Time available: ${datasetRow.minutes} minutes\n` +
      `Vibe: ${datasetRow.vibe}\n\n` +
      `Candidates:\n${candidates.map(describeGame).join('\n')}`;
    const client = weave.wrapOpenAI(inferenceClient());
    const response = await client.chat.completions.create({
      model: modelId(),
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: prompt },
      ],
    });
    return (response.choices[0].message.content ?? '').trim().replace(/^"|"$/g, '');
  },
  { name: 'GameNightModel' },
);

loadEnv();
await weave.init(weaveProject());

const evaluation = new weave.Evaluation({
  name: 'game-night-eval',
  dataset: new weave.Dataset({ name: 'game-night-scenarios', rows: SCENARIOS }),
  scorers: [validPick, fitsConstraints],
});

const summary = await evaluation.evaluate({ model: gameNightModel });
console.log(JSON.stringify(summary, null, 2));
