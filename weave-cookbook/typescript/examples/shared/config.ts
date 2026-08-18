// Environment and client configuration shared by the cookbook examples.

import OpenAI, { type ClientOptions } from 'openai';

export const INFERENCE_BASE_URL = 'https://api.inference.wandb.ai/v1';
export const DEFAULT_PROJECT = 'weave-cookbook';
export const DEFAULT_MODEL_ID = 'meta-llama/Llama-3.1-8B-Instruct';

export function loadEnv(): void {
  try {
    process.loadEnvFile();
  } catch {
    // No .env file; variables may be exported in the shell instead.
  }
}

export function requireApiKey(): string {
  const apiKey = process.env.WANDB_API_KEY;
  if (!apiKey) {
    console.error(
      'WANDB_API_KEY is not set. Copy .env.example to .env and add the key ' +
        'from https://wandb.ai/settings.',
    );
    process.exit(1);
  }
  return apiKey;
}

export function weaveProject(): string {
  const project = process.env.WANDB_PROJECT ?? DEFAULT_PROJECT;
  const entity = process.env.WANDB_ENTITY;
  return entity ? `${entity}/${project}` : project;
}

export function modelId(): string {
  return process.env.MODEL_ID ?? DEFAULT_MODEL_ID;
}

export function inferenceClient(): OpenAI {
  const options: ClientOptions = { baseURL: INFERENCE_BASE_URL, apiKey: requireApiKey() };
  if (process.env.WANDB_ENTITY) {
    options.project = weaveProject();
  }
  return new OpenAI(options);
}
