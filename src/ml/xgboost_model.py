"""XGBoost multiclass classifier for match outcome prediction.

Labels: H (home win), D (draw), A (away win).
After fit, classes_ is sorted alphabetically: ['A', 'D', 'H'].

Note: Model persistence uses pickle, as required by the training pipeline spec.
Only load models from trusted sources.
"""
from __future__ import annotations

import pickle  # noqa: S403 — required by spec; only load from trusted sources
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.config.settings import settings


@dataclass
class XGBoostMatchModel:
    """Multiclass XGBoost classifier: H (home win), D (draw), A (away win)."""

    params: dict = field(default_factory=lambda: {
        "n_estimators": settings.ml.n_estimators,
        "max_depth": settings.ml.max_depth,
        "learning_rate": settings.ml.learning_rate,
        "subsample": settings.ml.subsample,
        "colsample_bytree": settings.ml.colsample_bytree,
        "random_state": settings.ml.random_seed,
        "eval_metric": "mlogloss",
    })

    pipeline: Pipeline | None = field(default=None, repr=False)
    feature_names: list[str] | None = None
    classes_: list[str] | None = None  # ['A', 'D', 'H'] sorted alphabetically after fit

    # Internal: kept for inverse_transform
    _label_encoder: LabelEncoder | None = field(default=None, repr=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the pipeline.

        Parameters
        ----------
        X:
            Feature matrix (all columns must be numeric).
        y:
            String labels ('H', 'D', 'A').
        """
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        missing = set("ADH") - set(le.classes_)
        if missing:
            raise ValueError(
                f"Training data missing outcome class(es) {missing}. "
                "All three outcomes (H, D, A) must be represented."
            )

        self._label_encoder = le
        self.classes_ = list(le.classes_)  # ['A', 'D', 'H']
        self.feature_names = list(X.columns)

        xgb = XGBClassifier(**self.params)
        self.pipeline = Pipeline([("xgb", xgb)])
        self.pipeline.fit(X, y_encoded)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return (n_samples, 3) probability array.

        Columns are ordered: A, D, H (matching ``classes_`` alphabetical sort).
        """
        if self.pipeline is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self.pipeline.predict_proba(X)

    def predict_proba_dict(self, X: pd.DataFrame) -> list[dict[str, float]]:
        """Return list of dicts with keys 'home_win', 'draw', 'away_win' per row."""
        if self.classes_ is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        proba = self.predict_proba(X)  # shape (n_samples, 3)

        # Map alphabetical class indices to semantic keys
        idx_h = self.classes_.index("H")
        idx_d = self.classes_.index("D")
        idx_a = self.classes_.index("A")

        results = []
        for row in proba:
            results.append({
                "home_win": float(row[idx_h]),
                "draw": float(row[idx_d]),
                "away_win": float(row[idx_a]),
            })
        return results

    def save(self, path: Path) -> None:
        """Save model to pickle file. Creates parent directories if needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "XGBoostMatchModel":
        """Load a previously saved model from pickle file."""
        with open(Path(path), "rb") as f:
            obj = pickle.load(f)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected XGBoostMatchModel, got {type(obj)}")
        return obj
