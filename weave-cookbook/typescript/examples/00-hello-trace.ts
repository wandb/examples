// Log one conversation to Weave. No model call, no catalog -- just proof the wiring works.

import * as weave from 'weave';

try {
  process.loadEnvFile();
} catch {
  // No .env file; variables may be exported in the shell instead.
}

if (!process.env.WANDB_API_KEY) {
  console.error(
    'WANDB_API_KEY is not set. Copy .env.example to .env and add the key ' +
      'from https://wandb.ai/settings.',
  );
  process.exit(1);
}

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

await weave.flushOTel();
