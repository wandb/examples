import difflib
import json
from llama_cpp import Llama
import openai
import os
import re
import subprocess
import time
import wandb

config = {
    'max_tokens': int(os.getenv("MAX_TOKENS", 128)),
    'repetition_penalty': float(os.getenv("REPITITION_PENALTY", 1.1)), 
    'temperature': float(os.getenv("TEMP", 0.5)),
    'gpu_layers': int(os.getenv("GPU_LAYERS", 0)),
}

model_path = os.getenv("MODEL", "codellama-13b-instruct.Q4_K_M.gguf")
eval_path = os.getenv("EVAL_PATH", "eval.jsonl")
system_prompt = os.getenv("SYSTEM_PROMPT", "You're a Docker expert. Translate the following sentence to a simple docker command.")
diff_threshold = float(os.getenv("DIFF_THRESHOLD", 0.7))

def is_cuda_available():
    try:
        subprocess.check_output(["nvidia-smi"])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

wandb_config = {"model": model_path, "eval": eval_path,
                "system_prompt": system_prompt, **config}

# Set WANDB_MODE=disabled when running this files in tests
wandb.init(project="llm-eval-v2", config=wandb_config)

if wandb.config["model"].startswith("gpt"):
    def llm(prompt):
        res = openai.ChatCompletion.create(
            model=wandb.config["model"],
            messages=[
                {"role": "system", "content": wandb.config["system_prompt"]},
                {"role": "user", "content": prompt}
            ],
            temperature=wandb.config["temperature"],
            max_tokens=wandb.config["max_tokens"],
            frequency_penalty=wandb.config["repetition_penalty"],
        )
        return res.choices[0].message.content, res.usage.total_tokens
else:
    default_gpu = -1 if is_cuda_available() else 0
    cpp = Llama(f"/var/models/{wandb.config['model']}",
                verbose=bool(os.getenv("VERBOSE", False)),
                n_gpu_layers=int(os.getenv("GPU_LAYERS", default_gpu)))
    def llm(prompt):
        res = cpp.create_chat_completion(
            messages=[
                {"role": "system", "content": wandb.config["system_prompt"]},
                {"role": "user", "content": f"Q: {prompt}"}
            ],
            max_tokens=wandb.config["max_tokens"], stop=["Q:"],
            repeat_penalty=wandb.config["repetition_penalty"],
            temperature=wandb.config["temperature"],
        )
        return res["choices"][0]["message"]["content"], res["usage"]["total_tokens"]

print(f"Evaluating {wandb.config['model']}")
table = wandb.Table(columns=["prompt", "output", "ideal", "score", "latency", "tokens"])

codeblock_pattern = re.compile(r'(docker.+)$', re.MULTILINE)
def fmt(s):
    return f"`{s}`"

total_score = 0
total_latency = 0
total_tokens = 0
correct = 0.0
total = 0.0
with open(eval_path, "r") as f:
    for line in f:
        data = json.loads(line)
        total += 1.0
        prompt = data["input"]
        print(prompt)
        start = time.time()
        output, tokens = llm(prompt)
        latency = time.time() - start
        total_latency += latency
        matches = codeblock_pattern.findall(output)
        if len(matches) == 0:
            print("\t!!! No code generated:")
            for l in output.split("\n"):
                print(f"\t> {l}")
            continue
        command = matches[0].split("`")[0]
        score = difflib.SequenceMatcher(None, data["ideal"], command).ratio()
        print(f"\t({score:.2f}) {command}")
        total_score += score
        total_tokens += tokens
        if score > diff_threshold:
            correct += 1.0
        table.add_data(prompt, fmt(command), fmt(data["ideal"]), score, latency, tokens)

wandb.log({
    "accuracy": correct / total,
    "diff_score": total_score / total,
    "avg_tokens": total_tokens / total,
    "latency": total_latency / total,
    "eval": table
})
print("\nConfig:\n")
print(json.dumps(dict(wandb.config), indent=4))
print(f"Accuracy: {wandb.run.summary['accuracy']}")
print(f"Average diff score: {wandb.run.summary['diff_score']}")