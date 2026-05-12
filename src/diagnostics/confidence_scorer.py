from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd
from loguru import logger

from src.features.feature_builder import FEATURE_COLUMNS

_WEIGHTS: dict[str, float] = {
    "model_agreement": 0.35,
    "calibration_quality": 0.25,
    "feature_completeness": 0.20,
    "historical_reliability": 0.15,
    "prediction_volatility": 0.05,
}

_BANDS: list[tuple[str, float]] = [("High", 0.70), ("Medium", 0.40), ("Low", 0.0)]


def _shannon_entropy(probs: list[float]) -> float:
    """Shannon entropy (nats). Clips values to avoid log(0)."""
    return -sum(p * math.log(max(p, 1e-12)) for p in probs if p > 0)


class ConfidenceScorer:
    """Compute a [0, 1] confidence score for a single match prediction.

    The score is a weighted mean of 5 factors:
      - model_agreement (0.35)
      - calibration_quality (0.25)
      - feature_completeness (0.20)
      - historical_reliability (0.15)
      - prediction_volatility (0.05)
    """

    def __init__(self, matches_df: pd.DataFrame, calibration_ece: float) -> None:
        """
        Parameters
        ----------
        matches_df:
            Historical match DataFrame with columns: home_team, away_team.
        calibration_ece:
            Pre-computed Expected Calibration Error from benchmark results
            for the XGBoost model. Should be in [0, 1].
        """
        self._matches = matches_df
        self._calibration_ece = float(calibration_ece)
        self._team_match_counts = self._compute_match_counts()
        self._max_count = max(self._team_match_counts.values(), default=1)
        logger.debug(
            "ConfidenceScorer initialised: {} teams, max_count={}",
            len(self._team_match_counts),
            self._max_count,
        )

    def score(
        self,
        home_team: str,
        away_team: str,
        component_models: dict[str, dict[str, float]],
        ensemble_probs: dict[str, float],
        feature_vector: dict[str, float],
    ) -> dict:
        """Compute confidence score for a prediction.

        Parameters
        ----------
        home_team:
            Home team name.
        away_team:
            Away team name.
        component_models:
            Mapping of model name → probability dict, e.g.
            ``{"elo": {"home_win": p, "draw": p, "away_win": p}, ...}``.
            Expected keys: "elo", "poisson", "xgboost".
        ensemble_probs:
            Blended probability dict: ``{"home_win": p, "draw": p, "away_win": p}``.
        feature_vector:
            Dict of feature name → value (may contain None or NaN).

        Returns
        -------
        dict with keys:
            - confidence_score (float, 4 d.p.)
            - confidence_band (str: "High" | "Medium" | "Low")
            - factor_breakdown (dict of factor → float, 4 d.p.)
        """
        factors: dict[str, float] = {
            "model_agreement": self._model_agreement(component_models),
            "calibration_quality": max(0.0, 1.0 - self._calibration_ece),
            "feature_completeness": self._feature_completeness(feature_vector),
            "historical_reliability": self._historical_reliability(home_team, away_team),
            "prediction_volatility": self._prediction_volatility(ensemble_probs),
        }

        raw_score = sum(_WEIGHTS[k] * v for k, v in factors.items())
        final_score = max(0.0, min(1.0, raw_score))

        band = "Low"
        for band_name, threshold in _BANDS:
            if final_score >= threshold:
                band = band_name
                break

        logger.debug(
            "ConfidenceScorer: {} vs {} → score={:.4f} band={}",
            home_team,
            away_team,
            final_score,
            band,
        )

        return {
            "confidence_score": round(final_score, 4),
            "confidence_band": band,
            "factor_breakdown": {k: round(v, 4) for k, v in factors.items()},
        }

    # ------------------------------------------------------------------
    # Factor computations
    # ------------------------------------------------------------------

    def _model_agreement(self, component_models: dict[str, dict[str, float]]) -> float:
        """1 − mean(pairwise max absolute difference across outcomes)."""
        keys = ["home_win", "draw", "away_win"]
        model_vecs = {
            name: [probs[k] for k in keys]
            for name, probs in component_models.items()
        }
        pairs = list(combinations(model_vecs.values(), 2))
        if not pairs:
            return 1.0
        max_diffs = [
            max(abs(a - b) for a, b in zip(v1, v2))
            for v1, v2 in pairs
        ]
        return max(0.0, 1.0 - float(np.mean(max_diffs)))

    def _feature_completeness(self, feature_vector: dict[str, float]) -> float:
        """Fraction of FEATURE_COLUMNS that are non-null / non-NaN."""
        if not FEATURE_COLUMNS:
            return 1.0
        non_null = sum(
            1
            for col in FEATURE_COLUMNS
            if col in feature_vector
            and feature_vector[col] is not None
            and not (
                isinstance(feature_vector[col], float)
                and math.isnan(feature_vector[col])
            )
        )
        return non_null / len(FEATURE_COLUMNS)

    def _historical_reliability(self, home_team: str, away_team: str) -> float:
        """log(1 + combined_match_count) / log(1 + 2 * max_count), clamped to [0, 1]."""
        home_count = self._team_match_counts.get(home_team, 0)
        away_count = self._team_match_counts.get(away_team, 0)
        match_count = home_count + away_count
        denominator = math.log1p(self._max_count * 2)
        if denominator == 0.0:
            return 1.0
        raw = math.log1p(match_count) / denominator
        return max(0.0, min(1.0, raw))

    def _prediction_volatility(self, ensemble_probs: dict[str, float]) -> float:
        """1 − entropy(ensemble_probs) / log(3)."""
        probs = [
            ensemble_probs.get("home_win", 0.0),
            ensemble_probs.get("draw", 0.0),
            ensemble_probs.get("away_win", 0.0),
        ]
        entropy = _shannon_entropy(probs)
        max_entropy = math.log(3)
        return max(0.0, 1.0 - entropy / max_entropy)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_match_counts(self) -> dict[str, int]:
        """Count how many matches each team appears in (home or away)."""
        counts: dict[str, int] = {}
        for _, row in self._matches.iterrows():
            for team in [row["home_team"], row["away_team"]]:
                counts[team] = counts.get(team, 0) + 1
        return counts
