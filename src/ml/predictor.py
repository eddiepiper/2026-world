"""Inference: load a trained XGBoostMatchModel and predict single matches."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from src.features.feature_builder import FeatureBuilder
from src.ml.xgboost_model import XGBoostMatchModel


class MatchPredictor:
    """Load a trained XGBoostMatchModel and predict single matches."""

    def __init__(self, model: XGBoostMatchModel, feature_builder: FeatureBuilder) -> None:
        self._model = model
        self._feature_builder = feature_builder

    @classmethod
    def from_artifacts(cls, model_dir: Path, feature_builder: FeatureBuilder) -> "MatchPredictor":
        """Load model from model_dir/model.pkl and construct predictor."""
        model_path = Path(model_dir) / "model.pkl"
        logger.info("Loading model from {}", model_path)
        model = XGBoostMatchModel.load(model_path)
        return cls(model=model, feature_builder=feature_builder)

    def predict(
        self,
        home_team: str,
        away_team: str,
        match_date: pd.Timestamp | None = None,
    ) -> dict[str, float]:
        """Return {'home_win': float, 'draw': float, 'away_win': float}.

        Parameters
        ----------
        home_team:
            Name of the home team.
        away_team:
            Name of the away team.
        match_date:
            Date to use for rolling-form feature lookups.
            If None, today's date is used (features use all history up to now).
        """
        if match_date is None:
            match_date = pd.Timestamp.today().normalize()

        feature_dict = self._feature_builder.build_features_for_match(
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
        )

        # Build a single-row DataFrame with features in the correct column order
        X = pd.DataFrame([feature_dict])

        results = self._model.predict_proba_dict(X)
        probs = results[0]
        logger.debug(
            "Predict {} vs {} on {}: home_win={:.3f} draw={:.3f} away_win={:.3f}",
            home_team, away_team, match_date.date(),
            probs["home_win"], probs["draw"], probs["away_win"],
        )
        return probs
