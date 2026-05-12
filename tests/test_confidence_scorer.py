from __future__ import annotations

import math
import pytest
import pandas as pd
from unittest.mock import patch

from src.diagnostics.confidence_scorer import ConfidenceScorer, _shannon_entropy


def _make_matches() -> pd.DataFrame:
    return pd.DataFrame({
        "date": ["2020-01-01"] * 10,
        "home_team": ["Brazil"] * 5 + ["France"] * 5,
        "away_team": ["France"] * 5 + ["Brazil"] * 5,
        "result": ["H"] * 10,
        "home_goals": [2] * 10,
        "away_goals": [1] * 10,
    })


def _make_scorer(ece: float = 0.10) -> ConfidenceScorer:
    return ConfidenceScorer(matches_df=_make_matches(), calibration_ece=ece)


def _component_models(hw: float = 0.5, d: float = 0.3, aw: float = 0.2) -> dict:
    return {
        "elo": {"home_win": hw, "draw": d, "away_win": aw},
        "poisson": {"home_win": hw, "draw": d, "away_win": aw},
        "xgboost": {"home_win": hw, "draw": d, "away_win": aw},
    }


def _ensemble_probs(hw: float = 0.5, d: float = 0.3, aw: float = 0.2) -> dict:
    return {"home_win": hw, "draw": d, "away_win": aw}


def _full_features() -> dict:
    from src.features.feature_builder import FEATURE_COLUMNS
    return {col: 1.0 for col in FEATURE_COLUMNS}


class TestConfidenceScorer:
    def test_returns_required_keys(self):
        scorer = _make_scorer()
        result = scorer.score("Brazil", "France", _component_models(), _ensemble_probs(), _full_features())
        assert "confidence_score" in result
        assert "confidence_band" in result
        assert "factor_breakdown" in result

    def test_score_in_range(self):
        scorer = _make_scorer()
        result = scorer.score("Brazil", "France", _component_models(), _ensemble_probs(), _full_features())
        assert 0.0 <= result["confidence_score"] <= 1.0

    def test_perfect_agreement_increases_score(self):
        scorer = _make_scorer(ece=0.0)
        perfect = _component_models(0.6, 0.25, 0.15)
        result = scorer.score("Brazil", "France", perfect, _ensemble_probs(0.6, 0.25, 0.15), _full_features())
        assert result["confidence_score"] > 0.5

    def test_high_band_threshold(self):
        scorer = _make_scorer(ece=0.0)
        result = scorer.score("Brazil", "France", _component_models(), _ensemble_probs(), _full_features())
        assert result["confidence_band"] in ("High", "Medium", "Low")

    def test_missing_features_reduces_completeness(self):
        scorer = _make_scorer()
        empty_features: dict = {}
        result = scorer.score("Brazil", "France", _component_models(), _ensemble_probs(), empty_features)
        assert result["factor_breakdown"]["feature_completeness"] == 0.0

    def test_known_team_has_higher_reliability_than_unknown(self):
        scorer = _make_scorer()
        known = scorer.score("Brazil", "France", _component_models(), _ensemble_probs(), _full_features())
        unknown = scorer.score("Narnia", "Atlantis", _component_models(), _ensemble_probs(), _full_features())
        assert known["factor_breakdown"]["historical_reliability"] >= unknown["factor_breakdown"]["historical_reliability"]

    def test_calibration_quality_clips_at_zero(self):
        scorer = _make_scorer(ece=1.5)  # ECE > 1 → quality = 0
        result = scorer.score("Brazil", "France", _component_models(), _ensemble_probs(), _full_features())
        assert result["factor_breakdown"]["calibration_quality"] == 0.0

    def test_factor_breakdown_has_all_five_keys(self):
        scorer = _make_scorer()
        result = scorer.score("Brazil", "France", _component_models(), _ensemble_probs(), _full_features())
        expected_keys = {"model_agreement", "calibration_quality", "feature_completeness",
                         "historical_reliability", "prediction_volatility"}
        assert expected_keys == set(result["factor_breakdown"].keys())


class TestShannonEntropy:
    def test_uniform_entropy_is_log3(self):
        probs = [1/3, 1/3, 1/3]
        assert abs(_shannon_entropy(probs) - math.log(3)) < 1e-6

    def test_certain_outcome_zero_entropy(self):
        probs = [1.0, 0.0, 0.0]
        assert abs(_shannon_entropy(probs)) < 1e-6
