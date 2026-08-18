# Version objects

This guide writes objects to Weave but makes no model call. Any JSON-serializable value can be published as a named, versioned object with an immutable history ([objects docs](https://docs.wandb.ai/weave/guides/tracking/objects)). Datasets and prompts build on this same mechanism.

## How it works

In TypeScript, publishing and retrieval go through the client that `weave.init` returns: `client.publish(obj, name)` and `client.get(ref)`. [`examples/04-versioning.ts`](../../examples/04-versioning.ts) publishes the game catalog twice — first as-is, then with one game added — and retrieves both versions back using the `ObjectRef` values `publish` returned:

```typescript
const first = await client.publish(loadCatalog(), 'game-catalog');
console.log(`published: ${first.uri()}`);

const expanded = [...loadCatalog(), EXPANSION_GAME];
const second = await client.publish(expanded, 'game-catalog');
console.log(`published: ${second.uri()}`);

const original = (await client.get(first)) as Game[];
const latest = (await client.get(second)) as Game[];
```

Versions are content-addressed: publishing unchanged content creates no new version, and identical content yields the same digest — even from the Python SDK. Name-and-alias lookups like `game-catalog:v0` are Python-only today; in TypeScript, hold on to the `ObjectRef` from `publish` or rebuild one with `ObjectRef.fromUri(...)`.

## Run it

```bash
npm run versioning
```

## What you should see

```text
published: weave:///<entity>/weave-cookbook/object/game-catalog:<digest>
published: weave:///<entity>/weave-cookbook/object/game-catalog:<digest>
latest has 13 games; the first version still has 12.
```

Re-running without changing the catalog prints the same digests and creates no new versions.

## Inspect it in Weave

Open the printed link and switch to **Objects**:

- `game-catalog` shows two versions with row counts and digests.
- Select the first version, then the second, to see exactly what changed — the `Compost Wars` row.

Full capture: [typescript-04-versioning.txt](../../../assets/expected-output/typescript-04-versioning.txt).
