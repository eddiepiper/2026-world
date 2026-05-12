"""Ensemble engine — blends Elo, Poisson, and XGBoost probabilities.

This is the top-level prediction module for Phase 2. It combines three
component models using configurable weights and returns a rich prediction
dict with per-model breakdowns and a confidence score.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import pandas as pd

from src.config.settings import settings
from src.features.feature_builder import FeatureBuilder
from src.ml.calibration import ProbabilityCalibrator
from src.ml.xgboost_model import XGBoostMatchModel
from src.models.elo_model import EloModel
from src.models.poisson_model import PoissonModel
from src.utils.helpers import clamp, normalize_probabilities


@dataclass
class EnsembleWeights:
    """Configurable blend weights for the three component models.

    Weights must be non-negative and sum to approximately 1.0 (±1e-6).
    Defaults read from ``settings.ensemble``.
    """

    elo: float = field(default_factory=lambda: settings.ensemble.elo_weight)
    poisson: float = field(default_factory=lambda: settings.ensemble.poisson_weight)
    xgboost: float = field(default_factory=lambda: settings.ensemble.xgboost_weight)

    def validate(self) -> None:
        """Raise ValueError if weights don't sum to ~1.0 or any weight is negative."""
        if self.elo < 0 or self.poisson < 0 or self.xgboost < 0:
            raise ValueError(
                f"All ensemble weights must be non-negative. "
                f"Got elo={self.elo}, poisson={self.poisson}, xgboost={self.xgboost}."
            )
        total = self.elo + self.poisson + self.xgboost
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Ensemble weights must sum to 1.0, got {total:.8f}. "
                f"(elo={self.elo}, poisson={self.poisson}, xgboost={self.xgboost})"
            )


class EnsembleEngine:
    """Blends Elo + Poisson + XGBoost probabilities using configurable weights.

    Returns a rich prediction dict matching the spec output format::

        {
            "home_team": str,
            "away_team": str,
            "ensemble_probabilities": {
                "home_win": float,
                "draw": float,
                "away_win": float,
            },
            "component_models": {
                "elo":     {"home_win": float, "draw": float, "away_win": float},
                "poisson": {"home_win": float, "draw": float, "away_win": float},
                "xgboost": {"home_win": float, "draw": float, "away_win": float},
            },
            "confidence_score": float,
        }
    """

    def __init__(
        self,
        elo_model: EloModel,
        poisson_model: PoissonModel,
        xgb_model: XGBoostMatchModel | ProbabilityCalibrator,
        feature_builder: FeatureBuilder,
        weights: EnsembleWeights | None = None,
    ) -> None:
        """Initialise the engine.

        Parameters
        ----------
        elo_model:
            A trained ``EloModel`` instance.
        poisson_model:
            A trained ``PoissonModel`` instance.
        xgb_model:
            A fitted ``XGBoostMatchModel`` or ``ProbabilityCalibrator``.
            Both expose ``predict_proba_dict(X)``.
        feature_builder:
            A ``FeatureBuilder`` used to construct XGBoost input features.
        weights:
            Optional ``EnsembleWeights``. Defaults to ``EnsembleWeights()``
            (values from ``settings.ensemble``). Validated at construction.
        """
        self._elo = elo_model
        self._poisson = poisson_model
        self._xgb = xgb_model
        self._feature_builder = feature_builder

        self._weights = weights if weights is not None else EnsembleWeights()
        self._weights.validate()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        home_team: str,
        away_team: str,
        match_date: pd.Timestamp | None = None,
    ) -> dict:
        """Return the full prediction dict.

        Steps:

        1. Get Elo probabilities via ``elo_model.win_draw_loss_probabilities()``.
        2. Get Poisson probabilities via ``poisson_model.win_draw_loss_from_poisson()``.
        3. Build XGBoost features and get probabilities via
           ``xgb_model.predict_proba_dict()``.
        4. Compute weighted average of all three component models.
        5. Normalise ensemble probabilities to sum to 1.
        6. Compute ``confidence_score``.
        7. Return the full prediction dict.

        Parameters
        ----------
        home_team:
            Name of the home team.
        away_team:
            Name of the away team.
        match_date:
            Date of the match. Defaults to ``pd.Timestamp.today()`` when
            ``None``.

        Returns
        -------
        dict
            Full prediction dict with keys ``home_team``, ``away_team``,
            ``ensemble_probabilities``, ``component_models``, and
            ``confidence_score``.
        """
        if match_date is None:
            match_date = pd.Timestamp.today()

        # --- 1. Elo probabilities ---
        elo_win, elo_draw, elo_loss = self._elo.win_draw_loss_probabilities(
            home_team, away_team
        )
        elo_probs = {"home_win": elo_win, "draw": elo_draw, "away_win": elo_loss}

        # --- 2. Poisson probabilities ---
        poi_win, poi_draw, poi_loss = self._poisson.win_draw_loss_from_poisson(
            home_team, away_team
        )
        poisson_probs = {
            "home_win": poi_win,
            "draw": poi_draw,
            "away_win": poi_loss,
        }

        # --- 3. XGBoost probabilities ---
        features = self._feature_builder.build_features_for_match(
            home_team, away_team, match_date
        )
        feature_df = pd.DataFrame([features])
        xgb_result = self._xgb.predict_proba_dict(feature_df)
        xgb_probs = xgb_result[0]  # single match → first (and only) row

        # --- 4. Weighted average ---
        w = self._weights
        raw_ensemble = {
            "home_win": (
                w.elo * elo_probs["home_win"]
                + w.poisson * poisson_probs["home_win"]
                + w.xgboost * xgb_probs["home_win"]
            ),
            "draw": (
                w.elo * elo_probs["draw"]
                + w.poisson * poisson_probs["draw"]
                + w.xgboost * xgb_probs["draw"]
            ),
            "away_win": (
                w.elo * elo_probs["away_win"]
                + w.poisson * poisson_probs["away_win"]
                + w.xgboost * xgb_probs["away_win"]
            ),
        }

        # --- 5. Normalise ---
        norm_values = normalize_probabilities(
            [raw_ensemble["home_win"], raw_ensemble["draw"], raw_ensemble["away_win"]]
        )
        ensemble_probs = {
            "home_win": norm_values[0],
            "draw": norm_values[1],
            "away_win": norm_values[2],
        }

        # --- 6. Confidence score ---
        component_models = {
            "elo": elo_probs,
            "poisson": poisson_probs,
            "xgboost": xgb_probs,
        }
        confidence = self._confidence_score(ensemble_probs, component_models)

        # --- 7. Return full dict ---
        return {
            "home_team": home_team,
            "away_team": away_team,
            "ensemble_probabilities": ensemble_probs,
            "component_models": component_models,
            "confidence_score": confidence,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _confidence_score(
        self,
        ensemble: dict[str, float],  # noqa: ARG002 — reserved for future use
        components: dict[str, dict[str, float]],
    ) -> float:
        """Confidence score: how much the component models agree.

        Formula
        -------
        For each pair of component models, compute the maximum absolute
        difference across the three outcome probabilities::

            pairwise_disagreement(A, B) = max(
                |A["home_win"] - B["home_win"]|,
                |A["draw"]     - B["draw"]    |,
                |A["away_win"] - B["away_win"]|,
            )

        Then::

            confidence = 1 - mean(pairwise_disagreement for all pairs)

        Clamped to [0, 1].

        Parameters
        ----------
        ensemble:
            The blended ensemble probabilities (not used in the current
            formula but passed for potential future extensions).
        components:
            Dict mapping model name → probability dict.
        """
        model_names = list(components.keys())
        pairs = list(itertools.combinations(model_names, 2))

        if not pairs:
            return 1.0  # only one model → perfectly confident

        disagreements: list[float] = []
        for name_a, name_b in pairs:
            probs_a = components[name_a]
            probs_b = components[name_b]
            pairwise = max(
                abs(probs_a["home_win"] - probs_b["home_win"]),
                abs(probs_a["draw"] - probs_b["draw"]),
                abs(probs_a["away_win"] - probs_b["away_win"]),
            )
            disagreements.append(pairwise)

        mean_disagreement = sum(disagreements) / len(disagreements)
        return clamp(1.0 - mean_disagreement, 0.0, 1.0)
