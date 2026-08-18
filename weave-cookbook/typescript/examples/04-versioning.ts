// Publish, version, and retrieve objects: the catalog gets an immutable history.

import * as weave from 'weave';

import { loadCatalog, type Game } from './shared/catalog.js';
import { loadEnv, requireApiKey, weaveProject } from './shared/config.js';

const EXPANSION_GAME: Game = {
  name: 'Compost Wars',
  min_players: 3,
  max_players: 6,
  playtime_minutes: 35,
  difficulty: 'light',
  tags: ['party', 'engine-building'],
  description: 'Grow the mightiest compost heap before the first frost.',
};

loadEnv();
requireApiKey();
const client = await weave.init(weaveProject());

const first = await client.publish(loadCatalog(), 'game-catalog');
console.log(`published: ${first.uri()}`);

const expanded = [...loadCatalog(), EXPANSION_GAME];
const second = await client.publish(expanded, 'game-catalog');
console.log(`published: ${second.uri()}`);

const original = (await client.get(first)) as Game[];
const latest = (await client.get(second)) as Game[];
console.log(`latest has ${latest.length} games; the first version still has ${original.length}.`);
