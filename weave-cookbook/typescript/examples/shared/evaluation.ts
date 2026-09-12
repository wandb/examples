// Dataset and deterministic scorers shared by evaluation examples.

import * as weave from 'weave';

import { filterGames, loadCatalog } from './catalog.js';

export interface Scenario {
  players: number;
  minutes: number;
  vibe: string;
}

export const SCENARIOS: Scenario[] = [
  { players: 2, minutes: 30, vibe: 'quick and clever' },
  { players: 4, minutes: 60, vibe: 'cooperative' },
  { players: 5, minutes: 45, vibe: 'social deduction' },
  { players: 6, minutes: 90, vibe: 'epic and strategic' },
  // Nothing in the catalog fits this row; the correct answer is "none".
  { players: 2, minutes: 10, vibe: 'anything' },
];

export const validPick = weave.op(
  ({ modelOutput }: { modelOutput: string; datasetRow: Scenario }) => {
    const names = new Set(loadCatalog().map((game) => game.name));
    return { valid_pick: modelOutput === 'none' || names.has(modelOutput) };
  },
  { name: 'valid_pick' },
);

export const fitsConstraints = weave.op(
  ({ modelOutput, datasetRow }: { modelOutput: string; datasetRow: Scenario }) => {
    const fitting = new Set(
      filterGames(loadCatalog(), {
        players: datasetRow.players,
        maxMinutes: datasetRow.minutes,
      }).map((game) => game.name),
    );
    if (modelOutput === 'none') {
      return { fits_constraints: fitting.size === 0 };
    }
    return { fits_constraints: fitting.has(modelOutput) };
  },
  { name: 'fits_constraints' },
);
