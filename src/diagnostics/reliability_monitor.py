from __future__ import annotations

import pandas as pd
from loguru import logger

from src.diagnostics.confidence_scorer import ConfidenceScorer
from src.diagnostics.drift_detection import DriftDetector
from src.diagnostics.prediction_stability import StabilityAnalyzer


class ReliabilityMonitor:
    """Aggregates confidence, drift, and stability signals into a single report."""

    def __init__(
        self,
        confidence_scorer: ConfidenceScorer,
        drift_detector: DriftDetector,
        stability_analyzer: StabilityAnalyzer,
    ) -> None:
        self._confidence = confidence_scorer
        self._drift = drift_detector
        self._stability = stability_analyzer

    def report(
        self,
        home_team: str,
        away_team: str,
        match_date: pd.Timestamp | None,
        component_models: dict[str, dict[str, float]],
        ensemble_probs: dict[str, float],
        feature_vector: dict[str, float],
    ) -> dict:
        """Generate a combined reliability report for a match prediction.

        Returns:
            dict with keys: confidence, drift, stability
        """
        confidence = self._confidence.score(
            home_team=home_team,
            away_team=away_team,
            component_models=component_models,
            ensemble_probs=ensemble_probs,
            feature_vector=feature_vector,
        )
        drift = self._drift.status()
        stability = self._stability.analyze(
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
        )

        logger.debug(
            f"Reliability report: {home_team} vs {away_team} — "
            f"confidence={confidence['confidence_score']:.2f} "
            f"({confidence['confidence_band']}), "
            f"stability={stability['stability_band']}"
        )

        return {
            "confidence": confidence,
            "drift": drift,
            "stability": stability,
        }
