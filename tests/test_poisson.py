import pandas as pd
import pytest

from src.models.poisson_model import PoissonModel
from src.config.settings import PoissonConfig


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "date": pd.to_datetime(["2022-01-01"] * 10),
        "home_team": ["Brazil", "France", "Spain", "Germany", "England",
                      "Brazil", "France", "Spain", "Germany", "England"],
        "away_team": ["France", "Spain", "Germany", "England", "Brazil",
                      "Germany", "England", "Brazil", "France", "Spain"],
        "home_goals": [2, 1, 3, 1, 2, 1, 2, 0, 2, 3],
        "away_goals": [1, 1, 0, 2, 1, 0, 1, 2, 0, 1],
    })


@pytest.fixture
def trained_model(sample_df):
    model = PoissonModel()
    model.fit(sample_df)
    return model


def test_fit_sets_avg_goals(trained_model):
    assert trained_model.avg_goals > 0


def test_fit_sets_attack_strength_for_all_teams(trained_model):
    for team in ["Brazil", "France", "Spain", "Germany", "England"]:
        assert team in trained_model.attack_strength
        assert trained_model.attack_strength[team] > 0


def test_fit_sets_defense_strength_for_all_teams(trained_model):
    for team in ["Brazil", "France", "Spain", "Germany", "England"]:
        assert team in trained_model.defense_strength
        assert trained_model.defense_strength[team] > 0


def test_predict_goals_returns_positive(trained_model):
    home_xg, away_xg = trained_model.predict_goals("Brazil", "France")
    assert home_xg > 0
    assert away_xg > 0


def test_predict_goals_home_advantage_applied(trained_model):
    # With home_advantage=1.2, home xg should be inflated vs a neutral game
    model_neutral = PoissonModel(
        config=trained_model.config.__class__(
            home_advantage=1.0,
            default_attack=trained_model.config.default_attack,
            default_defense=trained_model.config.default_defense,
            avg_goals_fallback=trained_model.config.avg_goals_fallback,
        )
    )
    model_neutral.attack_strength = trained_model.attack_strength.copy()
    model_neutral.defense_strength = trained_model.defense_strength.copy()
    model_neutral.avg_goals = trained_model.avg_goals

    home_ha, _ = trained_model.predict_goals("Brazil", "France")
    home_no, _ = model_neutral.predict_goals("Brazil", "France")
    assert home_ha > home_no


def test_predict_goals_sane_range(trained_model):
    home_xg, away_xg = trained_model.predict_goals("Brazil", "France")
    assert 0.1 < home_xg < 10.0
    assert 0.1 < away_xg < 10.0


def test_simulate_score_non_negative(trained_model):
    h, a = trained_model.simulate_score("Brazil", "France")
    assert h >= 0
    assert a >= 0


def test_win_draw_loss_sums_to_one(trained_model):
    win, draw, loss = trained_model.win_draw_loss_from_poisson("Brazil", "France")
    assert abs(win + draw + loss - 1.0) < 0.02


def test_win_draw_loss_all_non_negative(trained_model):
    win, draw, loss = trained_model.win_draw_loss_from_poisson("Brazil", "France")
    assert win >= 0
    assert draw >= 0
    assert loss >= 0


def test_unknown_team_uses_default_strengths():
    model = PoissonModel()
    model.avg_goals = 1.5
    home_xg, away_xg = model.predict_goals("Unknown1", "Unknown2")
    assert home_xg > 0
    assert away_xg > 0


def test_fit_empty_dataframe():
    model = PoissonModel()
    df = pd.DataFrame(columns=["home_team", "away_team", "home_goals", "away_goals"])
    model.fit(df)
    # Should fall back to avg_goals_fallback
    assert model.avg_goals == model.config.avg_goals_fallback
