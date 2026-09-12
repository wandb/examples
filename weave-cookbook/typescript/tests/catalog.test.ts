import { describe, expect, it } from 'vitest';

import { describeGame, filterGames, loadCatalog } from '../examples/shared/catalog.js';

describe('catalog', () => {
  it('loads games with valid fields', () => {
    const games = loadCatalog();
    expect(games.length).toBeGreaterThanOrEqual(10);
    for (const game of games) {
      expect(game.min_players).toBeLessThanOrEqual(game.max_players);
      expect(['light', 'medium', 'heavy']).toContain(game.difficulty);
      expect(game.description.length).toBeGreaterThan(0);
    }
  });

  it('respects player bounds', () => {
    const games = filterGames(loadCatalog(), { players: 9, maxMinutes: 120 });
    expect(games.map((game) => game.name)).toEqual(['Whisper Network']);
  });

  it('respects the time budget', () => {
    const games = filterGames(loadCatalog(), { players: 4, maxMinutes: 60 });
    expect(games.map((game) => game.name)).toEqual([
      'Orchard Sprint',
      'Whisper Network',
      'Sky Ferry',
      "The Alchemist's Cellar",
      'Signal Lost',
      "Gemcutter's Guild",
    ]);
  });

  it('filters by difficulty', () => {
    const games = filterGames(loadCatalog(), { players: 2, maxMinutes: 180, difficulty: 'heavy' });
    expect(games.map((game) => game.name)).toEqual(['Moonbase Delta', 'Ironroot']);
  });

  it('returns empty when nothing fits', () => {
    expect(filterGames(loadCatalog(), { players: 4, maxMinutes: 10 })).toEqual([]);
  });

  it('describes a game on one line', () => {
    const line = describeGame(loadCatalog()[0]);
    expect(line.startsWith('- Orchard Sprint (3-8 players, 20 min, light):')).toBe(true);
    expect(line.includes('\n')).toBe(false);
  });
});
