// Load and filter the Game Night board game catalog.

import { readFileSync } from 'node:fs';

export type Difficulty = 'light' | 'medium' | 'heavy';

export interface Game {
  name: string;
  min_players: number;
  max_players: number;
  playtime_minutes: number;
  difficulty: Difficulty;
  tags: string[];
  description: string;
}

export interface FilterOptions {
  players: number;
  maxMinutes: number;
  difficulty?: Difficulty;
}

const CATALOG_URL = new URL('./data/games.json', import.meta.url);

export function loadCatalog(): Game[] {
  return JSON.parse(readFileSync(CATALOG_URL, 'utf8'));
}

export function filterGames(games: Game[], options: FilterOptions): Game[] {
  return games.filter(
    (game) =>
      game.min_players <= options.players &&
      options.players <= game.max_players &&
      game.playtime_minutes <= options.maxMinutes &&
      (options.difficulty === undefined || game.difficulty === options.difficulty),
  );
}

export function describeGame(game: Game): string {
  return (
    `- ${game.name} (${game.min_players}-${game.max_players} players, ` +
    `${game.playtime_minutes} min, ${game.difficulty}): ${game.description}`
  );
}
