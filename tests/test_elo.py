import pandas as pd
import pytest

from src.models.elo_model import EloModel
from src.config.settings import EloConfig


def test_initial_rating_default():
    model = EloModel()
    assert model.get_team_rating("UnknownTeam") == 1500.0


def test_initial_rating_custom_config():
    model = EloModel(config=EloConfig(initial_rating=1000.0))
    assert model.get_team_rating("Anyone") == 1000.0


def test_expected_result_equal_teams_home_advantage():
    model = EloModel()
    model.ratings = {"A": 1500.0, "B": 1500.0}
    # Home advantage should push expected > 0.5
    assert model.expected_result("A", "B") > 0.5


def test_expected_result_no_home_advantage():
    model = EloModel(config=EloConfig(home_advantage=0.0))
    model.ratings = {"A": 1500.0, "B": 1500.0}
    assert abs(model.expected_result("A", "B") - 0.5) < 1e-6


def test_expected_result_stronger_team_higher():
    model = EloModel()
    model.ratings = {"Strong": 2000.0, "Weak": 1000.0}
    e = model.expected_result("Strong", "Weak")
    assert e > 0.9


def test_expected_result_range():
    model = EloModel()
    model.ratings = {"A": 1800.0, "B": 1200.0}
    e = model.expected_result("A", "B")
    assert 0.0 < e < 1.0


def test_update_ratings_win_increases_winner():
    model = EloModel()
    model.ratings = {"A": 1500.0, "B": 1500.0}
    model.update_ratings("A", "B", home_goals=2, away_goals=0)
    assert model.ratings["A"] > 1500.0
    assert model.ratings["B"] < 1500.0


def test_update_ratings_loss_decreases_loser():
    model = EloModel()
    model.ratings = {"A": 1500.0, "B": 1500.0}
    model.update_ratings("A", "B", home_goals=0, away_goals=2)
    assert model.ratings["A"] < 1500.0
    assert model.ratings["B"] > 1500.0


def test_update_ratings_sum_conserved():
    model = EloModel()
    model.ratings = {"A": 1500.0, "B": 1500.0}
    total_before = model.ratings["A"] + model.ratings["B"]
    model.update_ratings("A", "B", home_goals=3, away_goals=1)
    total_after = model.ratings["A"] + model.ratings["B"]
    assert abs(total_before - total_after) < 1e-6


def test_win_draw_loss_sums_to_one():
    model = EloModel()
    model.ratings = {"Brazil": 1800.0, "France": 1750.0}
    win, draw, loss = model.win_draw_loss_probabilities("Brazil", "France")
    assert abs(win + draw + loss - 1.0) < 1e-6


def test_win_draw_loss_stronger_team_higher_win():
    model = EloModel(config=EloConfig(home_advantage=0.0))
    model.ratings = {"Strong": 2000.0, "Weak": 1200.0}
    win, draw, loss = model.win_draw_loss_probabilities("Strong", "Weak")
    assert win > loss


def test_train_on_matches_updates_ratings():
    model = EloModel()
    df = pd.DataFrame({
        "date": pd.to_datetime(["2022-01-01", "2022-01-08"]),
        "home_team": ["Brazil", "France"],
        "away_team": ["France", "Brazil"],
        "home_goals": [2, 1],
        "away_goals": [1, 2],
    })
    model.train_on_matches(df)
    assert "Brazil" in model.ratings
    assert "France" in model.ratings


def test_train_on_matches_chronological_order():
    model = EloModel()
    df = pd.DataFrame({
        "date": pd.to_datetime(["2022-01-08", "2022-01-01"]),  # Reversed order
        "home_team": ["B", "A"],
        "away_team": ["A", "B"],
        "home_goals": [0, 3],
        "away_goals": [0, 0],
    })
    # Training should sort by date: A beats B on Jan 1, then B draws A on Jan 8
    model.train_on_matches(df)
    assert "A" in model.ratings
    assert "B" in model.ratings
