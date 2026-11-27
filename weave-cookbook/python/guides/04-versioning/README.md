# Version objects

This guide writes objects to Weave but makes no model call. Any JSON-serializable value can be published as a named, versioned object with an immutable history ([objects docs](https://docs.wandb.ai/weave/guides/tracking/objects)). Datasets, models, and prompts build on this same mechanism.

## How it works

[`examples/04_versioning.py`](../../examples/04_versioning.py) publishes the game catalog twice — first as-is, then with one game added — and retrieves both versions back:

```python
    first = weave.publish(load_catalog(), name="game-catalog")
    print(f"published: {first.uri()}")

    expanded = load_catalog() + [EXPANSION_GAME]
    second = weave.publish(expanded, name="game-catalog")
    print(f"published: {second.uri()}")

    latest = weave.ref("game-catalog").get()
    original = weave.ref("game-catalog:v0").get()
    print(f"latest has {len(latest)} games; v0 still has {len(original)}.")
```

Three retrieval styles, all documented forms of the same ref:

- `weave.ref("game-catalog").get()` — the `:latest` alias.
- `weave.ref("game-catalog:v0").get()` — a pinned version.
- `weave.ref("weave:///<entity>/<project>/object/game-catalog:<digest>").get()` — fully qualified; works without `weave.init`.

Versions are content-addressed: publishing unchanged content creates no new version, and identical content yields the same digest — even from the TypeScript SDK.

## Run it

```bash
uv run python examples/04_versioning.py
```

## What you should see

```text
published: weave:///<entity>/weave-cookbook/object/game-catalog:<digest>
published: weave:///<entity>/weave-cookbook/object/game-catalog:<digest>
latest has 13 games; v0 still has 12.
```

Re-running without changing the catalog prints the same digests and creates no new versions.

## Inspect it in Weave

Open the printed link and switch to **Objects**:

- `game-catalog` shows two versions with row counts and digests.
- Select `v0`, then `v1`, to see exactly what changed — the `Compost Wars` row.

Full capture: [python-04-versioning.txt](../../../assets/expected-output/python-04-versioning.txt).
