from unittest.mock import MagicMock

import pytest

from src.models.prediction_engine import PredictionEngine, PredictionResult
from src.simulation.monte_carlo import MonteCarloSimulator


@pytest.fixture
def mock_engine():
    engine = MagicMock(spec=PredictionEngine)
    engine.predict.return_value = PredictionResult(
        home_team="A",
        away_team="B",
        win_probability=0.40,
        draw_probability=0.25,
        loss_probability=0.35,
        expected_home_goals=1.5,
        expected_away_goals=1.2,
    )
    return engine


@pytest.fixture
def sim(mock_engine):
    return MonteCarloSimulator(engine=mock_engine, n_simulations=200, random_seed=42)


# --- simulate_bracket ---

def test_simulate_bracket_winner_in_teams(sim):
    teams = ["Brazil", "France", "Spain", "Germany"]
    result = sim.simulate_bracket(teams)
    assert result["winner"] in teams


def test_simulate_bracket_two_teams(sim):
    result = sim.simulate_bracket(["A", "B"])
    assert result["winner"] in ["A", "B"]


def test_simulate_bracket_raises_on_single_team(sim):
    with pytest.raises(ValueError):
        sim.simulate_bracket(["OnlyOne"])


def test_simulate_bracket_finalists_tracked(sim):
    teams = ["A", "B", "C", "D"]
    result = sim.simulate_bracket(teams)
    assert len(result["finalists"]) == 2
    for t in result["finalists"]:
        assert t in teams


def test_simulate_bracket_semifinalists_tracked(sim):
    teams = ["A", "B", "C", "D"]
    result = sim.simulate_bracket(teams)
    assert len(result["semifinalists"]) == 4
    for t in result["semifinalists"]:
        assert t in teams


# --- run ---

def test_run_returns_dataframe(sim):
    df = sim.run(["Brazil", "France", "Spain", "Germany"])
    assert list(df.columns) == ["team", "winner_prob", "finalist_prob", "semifinal_prob"]


def test_run_includes_all_teams(sim):
    teams = ["Brazil", "France", "Spain", "Germany"]
    df = sim.run(teams)
    assert set(df["team"]) == set(teams)


def test_winner_probs_sum_to_one(sim):
    teams = ["Brazil", "France", "Spain", "Germany"]
    df = sim.run(teams)
    assert abs(df["winner_prob"].sum() - 1.0) < 0.05


def test_all_probabilities_non_negative(sim):
    teams = ["Brazil", "France", "Spain", "Germany"]
    df = sim.run(teams)
    assert (df["winner_prob"] >= 0).all()
    assert (df["finalist_prob"] >= 0).all()
    assert (df["semifinal_prob"] >= 0).all()


def test_finalist_prob_gte_winner_prob(sim):
    teams = ["A", "B", "C", "D", "E", "F", "G", "H"]
    df = sim.run(teams)
    for _, row in df.iterrows():
        assert row["finalist_prob"] >= row["winner_prob"] - 0.02


def test_semifinal_prob_gte_finalist_prob(sim):
    teams = ["A", "B", "C", "D", "E", "F", "G", "H"]
    df = sim.run(teams)
    for _, row in df.iterrows():
        assert row["semifinal_prob"] >= row["finalist_prob"] - 0.02


def test_sorted_by_winner_prob_descending(sim):
    teams = ["A", "B", "C", "D"]
    df = sim.run(teams)
    probs = df["winner_prob"].tolist()
    assert probs == sorted(probs, reverse=True)
