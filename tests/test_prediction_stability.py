from __future__ import annotations

import pandas as pd
import pytest
from unittest.mock import MagicMock

from src.diagnostics.prediction_stability import StabilityAnalyzer


def _make_analyzer(n_perturbations: int = 5) -> StabilityAnalyzer:
    from src.config.settings import DiagnosticsConfig
    cfg = DiagnosticsConfig()
    cfg.stability_n_perturbations = n_perturbations
    cfg.stability_noise_scale = 0.05

    elo = MagicMock()
    elo.win_draw_loss_probabilities.return_value = (0.5, 0.25, 0.25)

    poisson = MagicMock()
    poisson.win_draw_loss_from_poisson.return_value = (0.45, 0.30, 0.25)

    xgb = MagicMock()
    xgb.predict_proba_dict.return_value = [{"home_win": 0.5, "draw": 0.25, "away_win": 0.25}]

    feature_builder = MagicMock()
    feature_builder.build_features_for_match.return_value = {"elo_diff": 50.0, "home_form": 0.6}

    return StabilityAnalyzer(
        elo_model=elo,
        poisson_model=poisson,
        xgb_model=xgb,
        feature_builder=feature_builder,
        weights=(0.3, 0.3, 0.4),
        config=cfg,
        random_seed=42,
    )


class TestStabilityAnalyzer:
    def test_returns_required_keys(self):
        analyzer = _make_analyzer()
        result = analyzer.analyze("Brazil", "France")
        required = {"home_team", "away_team", "base_probabilities",
                    "mean_probabilities", "std_probabilities",
                    "stability_band", "n_perturbations"}
        assert required.issubset(result.keys())

    def test_home_team_away_team_in_result(self):
        analyzer = _make_analyzer()
        result = analyzer.analyze("Brazil", "France")
        assert result["home_team"] == "Brazil"
        assert result["away_team"] == "France"

    def test_n_perturbations_matches_config(self):
        analyzer = _make_analyzer(n_perturbations=7)
        result = analyzer.analyze("Brazil", "France")
        assert result["n_perturbations"] == 7

    def test_stability_band_is_valid(self):
        analyzer = _make_analyzer()
        result = analyzer.analyze("Brazil", "France")
        assert result["stability_band"] in ("Stable", "Moderate", "Unstable")

    def test_probabilities_sum_to_one(self):
        analyzer = _make_analyzer()
        result = analyzer.analyze("Brazil", "France")
        for prob_key in ("base_probabilities", "mean_probabilities"):
            probs = result[prob_key]
            total = probs["home_win"] + probs["draw"] + probs["away_win"]
            assert abs(total - 1.0) < 1e-3

    def test_std_probabilities_non_negative(self):
        analyzer = _make_analyzer()
        result = analyzer.analyze("Brazil", "France")
        for v in result["std_probabilities"].values():
            assert v >= 0.0

    def test_deterministic_with_seed(self):
        a1 = _make_analyzer()
        a2 = _make_analyzer()
        r1 = a1.analyze("Brazil", "France")
        r2 = a2.analyze("Brazil", "France")
        assert r1["mean_probabilities"] == r2["mean_probabilities"]

    def test_stable_band_when_xgb_always_same(self):
        """If XGBoost always returns the same output (mock), noise effect comes only via _perturb_features.
        With a constant mock, std should be very low → Stable."""
        analyzer = _make_analyzer(n_perturbations=10)
        result = analyzer.analyze("Brazil", "France")
        # XGBoost mock always returns same value; only feature perturbation differs
        # but mock ignores feature input, so std should be 0
        assert result["stability_band"] == "Stable"
