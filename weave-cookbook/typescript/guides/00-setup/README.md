# Setup

This cookbook traces a small Game Night Agent with [W&B Weave](https://weave-docs.wandb.ai/). Run everything from `typescript/` (Node.js 20.12+). This first guide logs one conversation — no model call yet, just proof the wiring works.

## How it works

`await weave.init(...)` connects the process to the `weave-cookbook` project, creating it on first use. `weave.startConversation` opens a conversation for `game-night-agent`, and `weave.startTurn` logs one user-agent exchange; the `finally` blocks close each span. From [`examples/00-hello-trace.ts`](../../examples/00-hello-trace.ts):

```typescript
const project = process.env.WANDB_PROJECT ?? 'weave-cookbook';
const entity = process.env.WANDB_ENTITY;
await weave.init(entity ? `${entity}/${project}` : project);

const conversation = weave.startConversation({ agentName: 'game-night-agent' });
try {
  const turn = weave.startTurn({ userMessage: 'Is game night still on?' });
  try {
    const reply = 'Welcome to game night! Bring snacks.';
    turn.record({ outputMessages: [{ role: 'assistant', content: reply }] });
    console.log(reply);
  } finally {
    turn.end();
  }
} finally {
  conversation.end();
}
```

The script ends with `await weave.flushOTel()`, which sends any buffered spans before the process exits.

## Run it

```bash
npm install
cp .env.example .env   # set WANDB_API_KEY from https://wandb.ai/settings
npm run hello
```

Logging to a team? Uncomment `WANDB_ENTITY` in `.env` and set the team name.

## What you should see

```text
View Weave data at https://wandb.ai/<entity>/weave-cookbook/weave
Welcome to game night! Bring snacks.
```

## Inspect it in Weave

Open the link the run prints and switch to **Agents**:

- One agent row, `game-night-agent`, with one conversation.
- Inside it, a single turn: user message `Is game night still on?` and the reply `Welcome to game night! Bring snacks.`

Full capture: [typescript-00-hello-trace.txt](../../../assets/expected-output/typescript-00-hello-trace.txt). Problems: [troubleshooting](../../../reference/TROUBLESHOOTING.md).
