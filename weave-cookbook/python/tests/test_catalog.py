from shared.catalog import describe_game, filter_games, load_catalog

REQUIRED_FIELDS = {
    "name",
    "min_players",
    "max_players",
    "playtime_minutes",
    "difficulty",
    "tags",
    "description",
}


def test_catalog_loads_with_required_fields():
    games = load_catalog()
    assert len(games) >= 10
    for game in games:
        assert REQUIRED_FIELDS <= set(game)
        assert game["min_players"] <= game["max_players"]
        assert game["difficulty"] in {"light", "medium", "heavy"}


def test_filter_respects_player_bounds():
    games = filter_games(load_catalog(), players=9, max_minutes=120)
    assert [game["name"] for game in games] == ["Whisper Network"]


def test_filter_respects_time_budget():
    games = filter_games(load_catalog(), players=4, max_minutes=60)
    names = [game["name"] for game in games]
    assert names == [
        "Orchard Sprint",
        "Whisper Network",
        "Sky Ferry",
        "The Alchemist's Cellar",
        "Signal Lost",
        "Gemcutter's Guild",
    ]


def test_filter_by_difficulty():
    games = filter_games(load_catalog(), players=2, max_minutes=180, difficulty="heavy")
    names = [game["name"] for game in games]
    assert names == ["Moonbase Delta", "Ironroot"]


def test_filter_returns_empty_when_nothing_fits():
    assert filter_games(load_catalog(), players=4, max_minutes=10) == []


def test_describe_game_is_one_line():
    line = describe_game(load_catalog()[0])
    assert line.startswith("- Orchard Sprint (3-8 players, 20 min, light):")
    assert "\n" not in line
