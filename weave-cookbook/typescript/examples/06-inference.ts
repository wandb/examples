// One key, many models: list them, switch with MODEL_ID, and tune response settings.

import * as weave from 'weave';

import { inferenceClient, loadEnv, modelId, weaveProject } from './shared/config.js';

const ANNOUNCER_PROMPT =
  "You are the Game Night Agent's announcer. Keep replies to one short sentence.";

loadEnv();
await weave.init(weaveProject());
const client = weave.wrapOpenAI(inferenceClient());

const models = await client.models.list();
console.log(`${models.data.length} models available through one API key; using ${modelId()}`);

const response = await client.chat.completions.create({
  model: modelId(),
  messages: [
    { role: 'system', content: ANNOUNCER_PROMPT },
    { role: 'user', content: 'Announce that game night starts in ten minutes.' },
  ],
  temperature: 0.2,
  max_tokens: 60,
});
console.log(response.choices[0].message.content);

const usage = response.usage;
console.log(
  usage
    ? `usage: ${usage.prompt_tokens} prompt + ${usage.completion_tokens} completion tokens`
    : 'usage: not reported',
);
