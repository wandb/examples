"""Examples must import and run their deterministic paths without any network access.

Each test loads the example module first and clears credentials afterward:
the examples call load_dotenv() at import, which would repopulate the
variables from a local .env file. Span APIs no-op without weave.init(),
so the deterministic paths run fully offline.
"""

import pytest
from conftest import load_example


def test_hello_trace_main_fails_actionably_without_key(monkeypatch):
    example = load_example("00_hello_trace.py")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="WANDB_API_KEY"):
        example.main()


def test_recommend_game_returns_early_when_nothing_fits(monkeypatch):
    example = load_example("01_trace_recommendation.py")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    reply = example.recommend_game(players=4, minutes=10, vibe="quick")
    assert reply == "none"


def test_follow_up_keeps_two_turns_in_one_conversation(monkeypatch):
    example = load_example("01_trace_recommendation.py")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    replies = example.run_conversation(
        players=4,
        minutes=10,
        vibe="quick",
        follow_up_minutes=5,
    )
    assert replies == ["none", "none"]


def test_recommend_game_fails_actionably_without_key(monkeypatch):
    example = load_example("01_trace_recommendation.py")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="WANDB_API_KEY"):
        example.recommend_game(players=4, minutes=60, vibe="friendly")


def test_valid_pick_accepts_catalog_games_and_none():
    example = load_example("02_evaluate.py")
    assert example.valid_pick(output="Tidepool") == {"valid_pick": True}
    assert example.valid_pick(output="none") == {"valid_pick": True}
    assert example.valid_pick(output="Monopoly") == {"valid_pick": False}


def test_fits_constraints_checks_players_and_minutes():
    example = load_example("02_evaluate.py")
    ok = example.fits_constraints(players=4, minutes=60, output="Gemcutter's Guild")
    assert ok == {"fits_constraints": True}
    bad = example.fits_constraints(players=4, minutes=60, output="Ironroot")
    assert bad == {"fits_constraints": False}
    impossible = example.fits_constraints(players=2, minutes=10, output="none")
    assert impossible == {"fits_constraints": True}


def test_model_predict_returns_none_when_nothing_fits(monkeypatch):
    example = load_example("02_evaluate.py")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    model = example.GameNightModel(model_name="unused", system_prompt="unused")
    assert model.predict(players=2, minutes=10, vibe="anything") == "none"


def test_eval_logger_batch_contains_a_failing_row():
    example = load_example("02_eval_logger.py")
    scores = [
        example.fits_constraints(row["players"], row["minutes"], row["output"])
        for row in example.BATCH
    ]
    assert scores == [True, True, False]


def test_judge_calibration_fixture_is_labeled_and_in_catalog():
    example = load_example("03_llm_judge.py")
    names = {game["name"] for game in example.load_catalog()}
    assert all(row["game"] in names for row in example.CALIBRATION)
    assert {row["human_says"] for row in example.CALIBRATION} == {True, False}


def test_agrees_with_human_compares_verdicts():
    example = load_example("03_llm_judge.py")
    yes = example.agrees_with_human(human_says=True, output={"vibe_match": True})
    assert yes == {"agrees_with_human": True}
    no = example.agrees_with_human(human_says=False, output={"vibe_match": True})
    assert no == {"agrees_with_human": False}


def test_judge_fails_actionably_without_key(monkeypatch):
    example = load_example("03_llm_judge.py")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="WANDB_API_KEY"):
        example.judge_vibe(vibe="cooperative", game="Moonbase Delta")


def test_expansion_game_matches_catalog_schema():
    example = load_example("04_versioning.py")
    reference = example.load_catalog()[0]
    assert set(example.EXPANSION_GAME) == set(reference)
    assert example.EXPANSION_GAME["difficulty"] in {"light", "medium", "heavy"}
    assert example.EXPANSION_GAME["min_players"] <= example.EXPANSION_GAME["max_players"]


def test_string_prompt_fills_the_vibe_placeholder():
    example = load_example("05_prompts.py")
    formatted = example.SYSTEM_PROMPT.format(vibe="cooperative")
    vibe_first = example.VIBE_FIRST_SYSTEM_PROMPT.format(vibe="cooperative")
    assert "cooperative" in formatted
    assert "{vibe}" not in formatted
    assert "cooperative" in vibe_first
    assert "{vibe}" not in vibe_first


def test_messages_prompt_fills_all_placeholders():
    example = load_example("05_prompts.py")
    messages = example.REQUEST_PROMPT.format(players=4, minutes=60, candidates="- Tidepool")
    assert messages[0]["role"] == "user"
    assert "Players: 4" in messages[0]["content"]
    assert "- Tidepool" in messages[0]["content"]


def test_inference_example_fails_actionably_without_key(monkeypatch):
    example = load_example("06_inference.py")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="WANDB_API_KEY"):
        example.main()


def test_otel_example_picks_the_quickest_fitting_game():
    from opentelemetry.trace import NoOpTracer

    example = load_example("08_otel.py")
    reply = example.recommend_game(NoOpTracer(), players=4, minutes=60)
    assert reply == "Play Orchard Sprint: done in 20 minutes."
    empty = example.recommend_game(NoOpTracer(), players=4, minutes=10)
    assert "No catalog game fits" in empty


def test_otel_example_fails_actionably_without_entity(monkeypatch):
    example = load_example("08_otel.py")
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    with pytest.raises(SystemExit, match="WANDB_ENTITY"):
        example.provider_for_weave()
