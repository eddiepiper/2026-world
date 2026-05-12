"""Tests for src/ensemble/ensemble_engine.py.

All component models are mocked so tests are fast and don't require real data.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ensemble.ensemble_engine import EnsembleEngine, EnsembleWeights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(
    elo_probs: tuple[float, float, float] = (0.5, 0.25, 0.25),
    poisson_probs: tuple[float, float, float] = (0.5, 0.25, 0.25),
    xgb_probs: dict[str, float] | None = None,
    weights: EnsembleWeights | None = None,
) -> EnsembleEngine:
    """Build an EnsembleEngine with mocked component models."""
    if xgb_probs is None:
        xgb_probs = {"home_win": 0.5, "draw": 0.25, "away_win": 0.25}

    elo_model = MagicMock()
    elo_model.win_draw_loss_probabilities.return_value = elo_probs

    poisson_model = MagicMock()
    poisson_model.win_draw_loss_from_poisson.return_value = poisson_probs

    xgb_model = MagicMock()
    xgb_model.predict_proba_dict.return_value = [xgb_probs]

    feature_builder = MagicMock()
    feature_builder.build_features_for_match.return_value = {"dummy_feature": 0.0}

    return EnsembleEngine(
        elo_model=elo_model,
        poisson_model=poisson_model,
        xgb_model=xgb_model,
        feature_builder=feature_builder,
        weights=weights,
    )


# ---------------------------------------------------------------------------
# EnsembleWeights validation
# ---------------------------------------------------------------------------

class TestEnsembleWeightsValidate:
    def test_validate_ok(self):
        """Valid weights that sum to 1.0 should not raise."""
        w = EnsembleWeights(elo=0.3, poisson=0.3, xgboost=0.4)
        w.validate()  # should not raise

    def test_validate_negative_raises(self):
        """Negative weight must raise ValueError."""
        w = EnsembleWeights(elo=-0.1, poisson=0.6, xgboost=0.5)
        with pytest.raises(ValueError, match="non-negative"):
            w.validate()

    def test_validate_sum_mismatch_raises(self):
        """Weights that don't sum to 1.0 must raise ValueError."""
        w = EnsembleWeights(elo=0.2, poisson=0.2, xgboost=0.2)
        with pytest.raises(ValueError, match="sum to 1.0"):
            w.validate()

    def test_validate_all_zero_raises(self):
        """All-zero weights don't sum to 1 and should raise."""
        w = EnsembleWeights(elo=0.0, poisson=0.0, xgboost=0.0)
        with pytest.raises(ValueError):
            w.validate()

    def test_validate_exactly_one_passes(self):
        """Single model weight = 1.0 with others = 0.0 is valid."""
        w = EnsembleWeights(elo=1.0, poisson=0.0, xgboost=0.0)
        w.validate()  # should not raise


# ---------------------------------------------------------------------------
# EnsembleEngine.predict — return structure
# ---------------------------------------------------------------------------

class TestEnsemblePredict:
    REQUIRED_TOP_KEYS = {
        "home_team",
        "away_team",
        "ensemble_probabilities",
        "component_models",
        "confidence_score",
    }
    OUTCOME_KEYS = {"home_win", "draw", "away_win"}

    def test_predict_returns_required_keys(self):
        engine = _make_engine()
        result = engine.predict("Brazil", "France")
        assert self.REQUIRED_TOP_KEYS == set(result.keys())

    def test_predict_probabilities_sum_to_one(self):
        engine = _make_engine()
        result = engine.predict("Brazil", "France")
        total = sum(result["ensemble_probabilities"].values())
        assert abs(total - 1.0) < 1e-9

    def test_predict_component_models_present(self):
        engine = _make_engine()
        result = engine.predict("Brazil", "France")
        components = result["component_models"]
        assert set(components.keys()) == {"elo", "poisson", "xgboost"}
        for model_probs in components.values():
            assert set(model_probs.keys()) == self.OUTCOME_KEYS

    def test_predict_uses_today_when_date_is_none(self):
        """When match_date=None the engine should call build_features_for_match
        with pd.Timestamp.today()."""
        engine = _make_engine()
        with patch("src.ensemble.ensemble_engine.pd.Timestamp") as mock_ts:
            fake_today = pd.Timestamp("2026-06-01")
            mock_ts.today.return_value = fake_today
            engine.predict("Brazil", "France", match_date=None)
            engine._feature_builder.build_features_for_match.assert_called_once_with(
                "Brazil", "France", fake_today
            )

    def test_predict_accepts_explicit_date(self):
        engine = _make_engine()
        match_date = pd.Timestamp("2026-06-15")
        result = engine.predict("Germany", "Argentina", match_date=match_date)
        assert result["home_team"] == "Germany"
        assert result["away_team"] == "Argentina"

    def test_predict_all_probabilities_in_range(self):
        engine = _make_engine(
            elo_probs=(0.6, 0.2, 0.2),
            poisson_probs=(0.55, 0.2, 0.25),
            xgb_probs={"home_win": 0.58, "draw": 0.22, "away_win": 0.20},
        )
        result = engine.predict("Brazil", "France")
        for p in result["ensemble_probabilities"].values():
            assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# confidence_score
# ---------------------------------------------------------------------------

class TestEnsembleConfidenceScore:
    def test_confidence_score_range(self):
        """Confidence must always be in [0, 1]."""
        engine = _make_engine()
        result = engine.predict("Brazil", "France")
        assert 0.0 <= result["confidence_score"] <= 1.0

    def test_confidence_score_high_when_models_agree(self):
        """When all three models return identical probabilities, confidence
        should be 1.0 (zero disagreement)."""
        probs = (0.5, 0.25, 0.25)
        xgb = {"home_win": 0.5, "draw": 0.25, "away_win": 0.25}
        engine = _make_engine(
            elo_probs=probs,
            poisson_probs=probs,
            xgb_probs=xgb,
        )
        result = engine.predict("Brazil", "France")
        assert result["confidence_score"] == pytest.approx(1.0, abs=1e-9)

    def test_confidence_score_lower_when_models_disagree(self):
        """When models differ significantly, confidence should be < 1.0."""
        engine = _make_engine(
            elo_probs=(0.9, 0.05, 0.05),
            poisson_probs=(0.1, 0.05, 0.85),
            xgb_probs={"home_win": 0.5, "draw": 0.25, "away_win": 0.25},
        )
        result = engine.predict("Brazil", "France")
        assert result["confidence_score"] < 1.0

    def test_confidence_score_direct(self):
        """Test _confidence_score directly with hand-crafted values."""
        engine = _make_engine()
        # Identical component models → disagreement = 0 → confidence = 1
        components = {
            "elo":     {"home_win": 0.6, "draw": 0.2, "away_win": 0.2},
            "poisson": {"home_win": 0.6, "draw": 0.2, "away_win": 0.2},
            "xgboost": {"home_win": 0.6, "draw": 0.2, "away_win": 0.2},
        }
        ensemble = {"home_win": 0.6, "draw": 0.2, "away_win": 0.2}
        score = engine._confidence_score(ensemble, components)
        assert score == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Weighted average correctness
# ---------------------------------------------------------------------------

class TestEnsembleWeightsApplied:
    def test_ensemble_weights_applied_correctly(self):
        """Manually compute the expected weighted average and compare."""
        elo_probs = (0.60, 0.20, 0.20)
        poisson_probs = (0.50, 0.30, 0.20)
        xgb_probs = {"home_win": 0.40, "draw": 0.35, "away_win": 0.25}

        weights = EnsembleWeights(elo=0.3, poisson=0.3, xgboost=0.4)
        engine = _make_engine(
            elo_probs=elo_probs,
            poisson_probs=poisson_probs,
            xgb_probs=xgb_probs,
            weights=weights,
        )
        result = engine.predict("Brazil", "France")

        # Expected weighted sum (before normalisation)
        expected_hw = 0.3 * 0.60 + 0.3 * 0.50 + 0.4 * 0.40
        expected_d  = 0.3 * 0.20 + 0.3 * 0.30 + 0.4 * 0.35
        expected_aw = 0.3 * 0.20 + 0.3 * 0.20 + 0.4 * 0.25
        total = expected_hw + expected_d + expected_aw

        ensemble = result["ensemble_probabilities"]
        assert ensemble["home_win"] == pytest.approx(expected_hw / total, abs=1e-9)
        assert ensemble["draw"]     == pytest.approx(expected_d  / total, abs=1e-9)
        assert ensemble["away_win"] == pytest.approx(expected_aw / total, abs=1e-9)

    def test_elo_only_weight(self):
        """With elo_weight=1.0, ensemble should match Elo probabilities exactly."""
        elo_probs = (0.70, 0.15, 0.15)
        poisson_probs = (0.40, 0.30, 0.30)
        xgb_probs = {"home_win": 0.33, "draw": 0.34, "away_win": 0.33}

        weights = EnsembleWeights(elo=1.0, poisson=0.0, xgboost=0.0)
        engine = _make_engine(
            elo_probs=elo_probs,
            poisson_probs=poisson_probs,
            xgb_probs=xgb_probs,
            weights=weights,
        )
        result = engine.predict("Brazil", "France")
        # After normalisation of a single-model output the values are identical
        total = sum(elo_probs)
        ensemble = result["ensemble_probabilities"]
        assert ensemble["home_win"] == pytest.approx(elo_probs[0] / total, abs=1e-9)
        assert ensemble["draw"]     == pytest.approx(elo_probs[1] / total, abs=1e-9)
        assert ensemble["away_win"] == pytest.approx(elo_probs[2] / total, abs=1e-9)
