# Overview

This script automates the evaluation of different LLM's with W&B.  It was originally used at Dockercon '23.  The default dataset attemtps to convert english commands into docker CLI commands.  See `eval.jsonl`.  All of the logic is in `evaluate.py`.

# Documentation

## Setup W&B

```bash
pip install wandb

# Find your api key at https://wandb.ai/authorize
export WANDB_API_KEY=XXX
# Find your openai api key at https://platform.openai.com/account/api-keys
export OPENAI_API_KEY=XXX
```

## Download Models

```bash
python download_models.py
```

## Nvidia/CUDA

### Build the docker container

```bash
docker build -t wandb/eval-llm:cuda .
```

### Run evaluation

```bash
docker run --gpus=all --cap-add SYS_RESOURCE -e USE_MLOCK=0 -e WANDB_API_KEY -e OPENAI_API_KEY -e MODEL=mistral-7b-instruct-v0.1.Q5_K_M.gguf -e TEMP=0.3 -v $(pwd)/models:/var/models wandb/eval-llm:cuda
```

## Environment variables

* `TEMP` - temperature _(0.5)_
* `MAX_TOKENS` - maximum number of tokens to emit _(128)_
* `SYSTEM_PROMPT` - instructions for the model _(You're a Docker expert. Translate the following sentence to a simple docker command.)_
* `MODEL` - name of gguf file, or gpt-turbo-3.5, gpt-40 _(codellama-13b-instruct.Q4_K_M.gguf)_
* `EVAL_PATH` - the path to a jsonl file with "input" and "ideal" keys _(eval.jsonl)_
* `VERBOSE` - print verbose info from llama-cpp-python _(False)_
* `DIFF_THRESHOLD` - the percentage threshold for considering a response accurate _(0.7)_
* `REPITITION_PENALTY` - how much to penalize repeated tokens _(1.1)_
* `GPU_LAYERS` - the number of layers to offload to the gpu _(-1 for CUDA, 0 for CPU)_

# W&B Launch Setup

## Create a queue

Goto https://wandb.ai/vanpelt/launch and create a queue named "llm-eval-cuda".  Set it's config to:

> Note: replace `/home/jupyter` to whatever `pwd` returns in your current directory.

```json
{
  "env": ["USE_MLOCK=0", "OPENAI_API_KEY"],
  "gpus": "all",
  "volume": "/home/jupyter/models:/var/models",
  "cap-add": "SYS_RESOURCE"
}
```

## Create a docker job

```bash
wandb job create --project "llm-eval" --name "llm-eval-cuda" image wandb/eval-llm:cuda
```

## Run an agent

```bash
wandb launch-agent -q llm-eval-cuda