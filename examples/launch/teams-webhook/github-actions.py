import wandb
import httpx
from tenacity import Retrying, stop_after_attempt, wait_random_exponential
import os

config = {
    "repo_owner": "andrewtruong",
    "repo": "wandb-gh-actions",
    "repo_ref": "master",
    "workflow_name": "manual-workflow.yml",
    "github_api_token_env_var": "GITHUB_API_TOKEN",
    "payload_inputs": {"template-file": "template.py"},
    "retry_settings": {
        "attempts": 3,
        "backoff": {"multiplier": 1, "max": 60},
    },
}

with wandb.init(config=config, job_type="github-actions-workflow-dispatch") as run:
    config = run.config

    base_url = "https://api.github.com"
    endpoint = "/repos/{repo_owner}/{repo}/actions/workflows/{workflow_name}/dispatches".format(
        **config
    )
    url = base_url + endpoint

    token = os.getenv(config["github_api_token_env_var"])
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    payload = {"ref": config["repo_ref"], "inputs": config["payload_inputs"]}

    for attempt in Retrying(
        stop=stop_after_attempt(config.retry_settings["attempts"]),
        wait=wait_random_exponential(**config.retry_settings["backoff"]),
    ):
        with attempt:
            httpx.post(url, headers=headers, json=payload)

    run.log_code()
