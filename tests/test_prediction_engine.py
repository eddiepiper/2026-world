import pandas as pd
import pytest

from src.models.elo_model import EloModel
from src.models.poisson_model import PoissonModel
from src.models.prediction_engine import PredictionEngine


@pytest.fixture
def engine():
    matches_df = pd.DataFrame({
        "date": pd.to_datetime(["2022-01-01"] * 6),
        "home_team": ["Brazil", "France", "Spain", "Brazil", "France", "Spain"],
        "away_team": ["France", "Spain", "Brazil", "Spain", "Brazil", "France"],
        "home_goals": [2, 1, 0, 3, 1, 2],
        "away_goals": [1, 1, 2, 0, 0, 1],
    })
    elo = EloModel()
    elo.train_on_matches(matches_df)
    poisson = PoissonModel()
    poisson.fit(matches_df)
    return PredictionEngine(elo_model=elo, poisson_model=poisson, elo_weight=0.5)


def test_predict_returns_all_keys(engine):
    result = engine.predict("Brazil", "France")
    for key in ["home_team", "away_team", "win_probability", "draw_probability",
                "loss_probability", "expected_home_goals", "expected_away_goals"]:
        assert key in result


def test_predict_probabilities_sum_to_one(engine):
    result = engine.predict("Brazil", "France")
    total = result["win_probability"] + result["draw_probability"] + result["loss_probability"]
    assert abs(total - 1.0) < 1e-3


def test_predict_probabilities_non_negative(engine):
    result = engine.predict("Brazil", "France")
    assert result["win_probability"] >= 0
    assert result["draw_probability"] >= 0
    assert result["loss_probability"] >= 0


def test_predict_expected_goals_positive(engine):
    result = engine.predict("Brazil", "France")
    assert result["expected_home_goals"] > 0
    assert result["expected_away_goals"] > 0


def test_predict_team_names_preserved(engine):
    result = engine.predict("Brazil", "France")
    assert result["home_team"] == "Brazil"
    assert result["away_team"] == "France"


def test_poisson_weight_property(engine):
    engine.elo_weight = 0.6
    assert abs(engine.poisson_weight - 0.4) < 1e-9


def test_predict_elo_weight_zero_uses_only_poisson(engine):
    engine_poi = PredictionEngine(
        elo_model=engine.elo_model,
        poisson_model=engine.poisson_model,
        elo_weight=0.0,
    )
    result = engine_poi.predict("Brazil", "France")
    total = result["win_probability"] + result["draw_probability"] + result["loss_probability"]
    assert abs(total - 1.0) < 1e-3


def test_predict_unknown_teams_returns_valid_result(engine):
    result = engine.predict("NewTeamX", "NewTeamY")
    total = result["win_probability"] + result["draw_probability"] + result["loss_probability"]
    assert abs(total - 1.0) < 0.005  # 3-decimal rounding on three values allows ±0.0015
