from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.optimization.ensemble_optimizer import WeightOptimizer


def _make_matches() -> pd.DataFrame:
    """70 matches: 20 before split, 50 on/after split date."""
    pre = pd.DataFrame({
        "date": ["2019-01-01"] * 20,
        "home_team": ["Brazil"] * 20,
        "away_team": ["France"] * 20,
        "result": ["H"] * 20,
        "home_goals": [2] * 20,
        "away_goals": [1] * 20,
    })
    post = pd.DataFrame({
        "date": ["2020-06-01"] * 50,
        "home_team": ["Brazil"] * 50,
        "away_team": ["France"] * 50,
        "result": (["H"] * 20 + ["D"] * 15 + ["A"] * 15),
        "home_goals": [2] * 50,
        "away_goals": [1] * 50,
    })
    return pd.concat([pre, post], ignore_index=True)


def _make_optimizer(matches_df: pd.DataFrame) -> WeightOptimizer:
    elo = MagicMock()
    elo.win_draw_loss_probabilities.return_value = (0.5, 0.25, 0.25)

    poisson = MagicMock()
    poisson.win_draw_loss_from_poisson.return_value = (0.45, 0.30, 0.25)

    xgb = MagicMock()
    xgb.predict_proba_dict.return_value = [{"home_win": 0.5, "draw": 0.25, "away_win": 0.25}]

    feature_builder = MagicMock()
    feature_builder.build_features_for_match.return_value = {}

    return WeightOptimizer(
        matches_df=matches_df,
        elo_model=elo,
        poisson_model=poisson,
        xgb_model=xgb,
        feature_builder=feature_builder,
        test_split_date="2020-01-01",
    )


class TestWeightOptimizer:
    def test_optimize_returns_required_keys(self):
        df = _make_matches()
        opt = _make_optimizer(df)
        result = opt.optimize()
        required = {"elo", "poisson", "xgboost", "optimized_log_loss",
                    "baseline_log_loss", "improvement", "method", "validated_at"}
        assert required.issubset(result.keys())

    def test_weights_sum_to_one(self):
        df = _make_matches()
        opt = _make_optimizer(df)
        result = opt.optimize()
        total = result["elo"] + result["poisson"] + result["xgboost"]
        assert abs(total - 1.0) < 1e-4

    def test_weights_non_negative(self):
        df = _make_matches()
        opt = _make_optimizer(df)
        result = opt.optimize()
        assert result["elo"] >= 0
        assert result["poisson"] >= 0
        assert result["xgboost"] >= 0

    def test_method_is_slsqp(self):
        df = _make_matches()
        opt = _make_optimizer(df)
        result = opt.optimize()
        assert result["method"] == "SLSQP"

    def test_save_writes_json(self, tmp_path):
        df = _make_matches()
        opt = _make_optimizer(df)
        result = opt.optimize()
        out = tmp_path / "models" / "best_weight_config.json"
        opt.save(out, result)
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["method"] == "SLSQP"

    def test_no_test_data_raises(self):
        df = pd.DataFrame({
            "date": ["2019-01-01"] * 5,
            "home_team": ["Brazil"] * 5,
            "away_team": ["France"] * 5,
            "result": ["H"] * 5,
            "home_goals": [2] * 5,
            "away_goals": [1] * 5,
        })
        opt = _make_optimizer(df)
        with pytest.raises(ValueError, match="No test data"):
            opt.optimize()
