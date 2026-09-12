// An LLM judge for the one property code can't measure: does the pick match the vibe?

import * as weave from 'weave';

import { describeGame, filterGames, loadCatalog, type Game } from './shared/catalog.js';
import { inferenceClient, loadEnv, modelId, weaveProject } from './shared/config.js';
import { isDirectExecution } from './shared/runtime.js';

type CalibrationRow = { vibe: string; game: string; human_says: boolean };
type Scenario = { players: number; minutes: number; vibe: string };
type Verdict = { vibe_match: boolean; reason: string };

const RUBRIC =
  'You judge board game recommendations. Decide only whether the game matches the ' +
  'requested vibe. Judge mood and play style; ignore player count and duration. ' +
  'Return vibe_match and a one-sentence reason.';

const VERDICT_FORMAT = {
  type: 'json_schema' as const,
  json_schema: {
    name: 'VibeVerdict',
    strict: true,
    schema: {
      type: 'object',
      properties: {
        vibe_match: { type: 'boolean' },
        reason: { type: 'string' },
      },
      required: ['vibe_match', 'reason'],
      additionalProperties: false,
    },
  },
};

// Human-labeled rows. The judge has to agree with these before its scores mean anything.
export const CALIBRATION: CalibrationRow[] = [
  { vibe: 'cooperative', game: 'Moonbase Delta', human_says: true },
  { vibe: 'cooperative', game: 'Caravan Kings', human_says: false },
  { vibe: 'quick party fun', game: 'Orchard Sprint', human_says: true },
  { vibe: 'quick party fun', game: 'Ironroot', human_says: false },
  { vibe: 'brainy deduction', game: "The Alchemist's Cellar", human_says: true },
  { vibe: 'brainy deduction', game: 'Sky Ferry', human_says: false },
];

const SCENARIOS: Scenario[] = [
  { players: 4, minutes: 60, vibe: 'cooperative' },
  { players: 2, minutes: 45, vibe: 'brainy deduction' },
  { players: 5, minutes: 30, vibe: 'quick party fun' },
];

const SYSTEM_PROMPT =
  'You are the Game Night Agent. Pick the best game for the group from the candidate list. ' +
  'Reply with the game name only, exactly as written in the list.';

function describe(gameName: string): string {
  const game = loadCatalog().find((entry: Game) => entry.name === gameName);
  return game ? describeGame(game) : gameName;
}

// Narrow LLM judge: one question, structured output.
const judgeVibe = weave.op(
  async function judge_vibe(vibe: string, game: string): Promise<Verdict> {
    const response = await weave.wrapOpenAI(inferenceClient()).chat.completions.create({
      model: modelId(),
      messages: [
        { role: 'system', content: RUBRIC },
        { role: 'user', content: `Requested vibe: ${vibe}\nRecommended game: ${describe(game)}` },
      ],
      response_format: VERDICT_FORMAT,
    });
    const content = response.choices[0].message.content;
    if (!content) {
      throw new Error('Judge returned no structured verdict.');
    }
    const verdict = JSON.parse(content) as Partial<Verdict>;
    if (typeof verdict.vibe_match !== 'boolean' || typeof verdict.reason !== 'string') {
      throw new Error('Judge verdict did not match the expected schema.');
    }
    return verdict as Verdict;
  },
);

const judgeModel = weave.op(
  async ({ datasetRow }: { datasetRow: CalibrationRow }) =>
    judgeVibe(datasetRow.vibe, datasetRow.game),
  { name: 'judge_vibe_model' },
);

const agreesWithHuman = weave.op(
  ({ modelOutput, datasetRow }: { modelOutput: Verdict; datasetRow: CalibrationRow }) => {
    return { agrees_with_human: modelOutput.vibe_match === datasetRow.human_says };
  },
  { name: 'agrees_with_human' },
);

// Scorer that delegates to the judge. Not ground truth -- a calibrated opinion.
const vibeMatch = weave.op(
  async ({ modelOutput, datasetRow }: { modelOutput: string; datasetRow: Scenario }) => {
    const verdict = await judgeVibe(datasetRow.vibe, modelOutput);
    return { vibe_match: verdict.vibe_match, judge_reason: verdict.reason };
  },
  { name: 'vibe_match' },
);

const recommend = weave.op(
  async ({ datasetRow }: { datasetRow: Scenario }) => {
    const candidates = filterGames(loadCatalog(), {
      players: datasetRow.players,
      maxMinutes: datasetRow.minutes,
    });
    if (candidates.length === 0) {
      return 'none';
    }
    const prompt =
      `Players: ${datasetRow.players}\n` +
      `Time available: ${datasetRow.minutes} minutes\n` +
      `Vibe: ${datasetRow.vibe}\n\n` +
      `Candidates:\n${candidates.map(describeGame).join('\n')}`;
    const response = await weave.wrapOpenAI(inferenceClient()).chat.completions.create({
      model: modelId(),
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: prompt },
      ],
    });
    return (response.choices[0].message.content ?? '').trim().replace(/^"|"$/g, '');
  },
  { name: 'recommend' },
);

export async function main(): Promise<void> {
  loadEnv();
  await weave.init(weaveProject());

  const calibration = new weave.Evaluation({
    name: 'vibe-judge-calibration',
    dataset: new weave.Dataset({ name: 'vibe-judge-calibration-rows', rows: CALIBRATION }),
    scorers: [agreesWithHuman],
  });
  const summary = await calibration.evaluate({ model: judgeModel });
  const agreement = summary.agrees_with_human.agrees_with_human.true_fraction;
  console.log(
    `Judge agrees with the human labels on ${Math.round(agreement * 100)}% of calibration rows.`,
  );

  const agentEval = new weave.Evaluation({
    name: 'vibe-judge-eval',
    dataset: new weave.Dataset({ name: 'vibe-scenarios', rows: SCENARIOS }),
    scorers: [vibeMatch],
  });
  const agentSummary = await agentEval.evaluate({ model: recommend });
  console.log(agentSummary.vibe_match.vibe_match);
}

if (isDirectExecution(import.meta.url)) {
  await main();
}
