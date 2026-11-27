"""Opt-in live checks: RUN_LIVE_TESTS=1 uv run pytest -q -m live"""

import os

import pytest
from conftest import load_example

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("RUN_LIVE_TESTS") != "1",
        reason="live tests are opt-in; set RUN_LIVE_TESTS=1",
    ),
]


def test_recommendation_produces_a_conversation():
    import weave

    from shared.config import weave_project

    example = load_example("01_trace_recommendation.py")
    weave.init(weave_project(), settings={"implicitly_patch_integrations": False})
    with weave.start_conversation(agent_name="game-night-agent"):
        reply = example.recommend_game(players=4, minutes=60, vibe="cooperative")

    assert reply.strip()
