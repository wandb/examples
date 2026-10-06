"""Environment and client configuration shared by the cookbook examples."""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

INFERENCE_BASE_URL = "https://api.inference.wandb.ai/v1"
DEFAULT_PROJECT = "weave-cookbook"
DEFAULT_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


class MissingConfigError(SystemExit):
    """Raised when a required environment variable is absent."""


def weave_project() -> str:
    """Return the Weave project path, prefixed with the entity when one is set."""
    project = os.getenv("WANDB_PROJECT", DEFAULT_PROJECT)
    entity = os.getenv("WANDB_ENTITY")
    return f"{entity}/{project}" if entity else project


def model_id() -> str:
    return os.getenv("MODEL_ID", DEFAULT_MODEL_ID)


def require_api_key() -> str:
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        raise MissingConfigError(
            "WANDB_API_KEY is not set. Copy .env.example to .env and add the key "
            "from https://wandb.ai/settings."
        )
    return api_key


def inference_client() -> OpenAI:
    """Build an OpenAI-compatible client pointed at W&B Serverless Inference."""
    kwargs: dict = {"base_url": INFERENCE_BASE_URL, "api_key": require_api_key()}
    if os.getenv("WANDB_ENTITY"):
        kwargs["project"] = weave_project()
    return OpenAI(**kwargs)
