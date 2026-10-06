// Game Night Agent: one conversation with traced tool and LLM work per turn.

import { parseArgs } from 'node:util';
import * as weave from 'weave';

import { describeGame, filterGames, loadCatalog } from './shared/catalog.js';
import { inferenceClient, loadEnv, modelId, weaveProject } from './shared/config.js';
import { isDirectExecution } from './shared/runtime.js';

const SYSTEM_PROMPT =
  'You are the Game Night Agent. Pick exactly one game from the candidate list. ' +
  'Reply with the game name only, exactly as written in the list. ' +
  'Never invent a game that is not on the list.';

export async function recommendGame(
  players: number,
  minutes: number,
  vibe: string,
  userMessage?: string,
): Promise<string> {
  const request =
    userMessage ?? `Recommend a game for ${players} players, ${minutes} minutes, vibe: ${vibe}.`;
  const turn = weave.startTurn({ userMessage: request });
  try {
    const tool = weave.startTool({
      name: 'search_catalog',
      args: JSON.stringify({ players, max_minutes: minutes }),
    });
    let candidates;
    try {
      candidates = filterGames(loadCatalog(), { players, maxMinutes: minutes });
      tool.result = JSON.stringify(candidates.map((game) => game.name));
    } finally {
      tool.end();
    }

    let reply: string;
    if (candidates.length === 0) {
      reply = 'none';
    } else {
      const prompt =
        `Players: ${players}\n` +
        `Time available: ${minutes} minutes\n` +
        `Vibe: ${vibe}\n\n` +
        `Candidates:\n${candidates.map(describeGame).join('\n')}`;
      const llm = weave.startLLM({ model: modelId(), providerName: 'openai' });
      try {
        const response = await inferenceClient().chat.completions.create({
          model: modelId(),
          messages: [
            { role: 'system', content: SYSTEM_PROMPT },
            { role: 'user', content: prompt },
          ],
        });
        reply = (response.choices[0].message.content ?? '').trim().replace(/^"|"$/g, '');
        llm.record({
          inputMessages: [
            { role: 'system', content: SYSTEM_PROMPT },
            { role: 'user', content: prompt },
          ],
          outputMessages: [{ role: 'assistant', content: reply }],
          usage: {
            inputTokens: response.usage?.prompt_tokens,
            outputTokens: response.usage?.completion_tokens,
          },
        });
      } finally {
        llm.end();
      }
    }

    turn.record({ outputMessages: [{ role: 'assistant', content: reply }] });
    return reply;
  } finally {
    turn.end();
  }
}

export async function runConversation(
  players: number,
  minutes: number,
  vibe: string,
  followUpMinutes?: number,
): Promise<string[]> {
  const replies: string[] = [];
  const conversation = weave.startConversation({
    agentName: 'game-night-agent',
    model: modelId(),
  });
  try {
    replies.push(await recommendGame(players, minutes, vibe));
    if (followUpMinutes !== undefined) {
      replies.push(
        await recommendGame(
          players,
          followUpMinutes,
          vibe,
          `What if we only have ${followUpMinutes} minutes?`,
        ),
      );
    }
  } finally {
    conversation.end();
  }
  return replies;
}

export async function main(): Promise<void> {
  loadEnv();
  const { values } = parseArgs({
    options: {
      players: { type: 'string', default: '4' },
      minutes: { type: 'string', default: '60' },
      vibe: { type: 'string', default: 'friendly and fun' },
      'follow-up-minutes': { type: 'string' },
    },
  });

  await weave.init(weaveProject());

  const followUpMinutes =
    values['follow-up-minutes'] === undefined
      ? undefined
      : Number(values['follow-up-minutes']);
  const replies = await runConversation(
    Number(values.players),
    Number(values.minutes),
    values.vibe,
    followUpMinutes,
  );
  for (const reply of replies) {
    console.log(reply);
  }

  await weave.flushOTel();
}

if (isDirectExecution(import.meta.url)) {
  await main();
}
