// Log an evaluation you already ran: no Evaluation object, one prediction at a time.

import * as weave from 'weave';
import { EvaluationLogger } from 'weave';

import { filterGames, loadCatalog } from './shared/catalog.js';
import { loadEnv, requireApiKey, weaveProject } from './shared/config.js';

// Outputs captured from an earlier batch run of the agent.
const BATCH = [
  { players: 2, minutes: 30, output: 'Tidepool' },
  { players: 4, minutes: 60, output: "Gemcutter's Guild" },
  // A bad pick, kept on purpose: Ironroot runs 120 minutes.
  { players: 3, minutes: 45, output: 'Ironroot' },
];

function fitsConstraints(players: number, minutes: number, output: string): boolean {
  const fitting = new Set(
    filterGames(loadCatalog(), { players, maxMinutes: minutes }).map((game) => game.name),
  );
  return fitting.has(output);
}

loadEnv();
requireApiKey();
await weave.init(weaveProject());

const evalLogger = new EvaluationLogger({
  name: 'game-night-batch',
  model: { name: 'game-night-agent' },
  dataset: 'batch-2026-08-02',
});

for (const row of BATCH) {
  const prediction = evalLogger.logPrediction(
    { players: row.players, minutes: row.minutes },
    row.output,
  );
  prediction.logScore('fits_constraints', fitsConstraints(row.players, row.minutes, row.output));
  prediction.finish();
}

await evalLogger.logSummary();
console.log(`Logged ${BATCH.length} predictions to evaluation 'game-night-batch'.`);
