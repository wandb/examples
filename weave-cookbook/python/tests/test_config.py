import pytest

from shared import config


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT", "MODEL_ID"):
        monkeypatch.delenv(name, raising=False)


def test_weave_project_defaults():
    assert config.weave_project() == "weave-cookbook"


def test_weave_project_includes_entity(monkeypatch):
    monkeypatch.setenv("WANDB_ENTITY", "my-team")
    monkeypatch.setenv("WANDB_PROJECT", "game-night")
    assert config.weave_project() == "my-team/game-night"


def test_model_id_defaults_and_overrides(monkeypatch):
    assert config.model_id() == "meta-llama/Llama-3.1-8B-Instruct"
    monkeypatch.setenv("MODEL_ID", "openai/gpt-oss-20b")
    assert config.model_id() == "openai/gpt-oss-20b"


def test_inference_client_fails_with_actionable_message():
    with pytest.raises(config.MissingConfigError, match="WANDB_API_KEY"):
        config.inference_client()


def test_inference_client_builds_when_key_is_set(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "test-key-not-real")
    client = config.inference_client()
    assert str(client.base_url).startswith("https://api.inference.wandb.ai")
