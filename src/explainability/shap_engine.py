from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

from src.features.feature_builder import FEATURE_COLUMNS
from src.ml.xgboost_model import XGBoostMatchModel

_SHAP_UNAVAILABLE_MSG = (
    "Explainability unavailable: shap is not installed. "
    "Install it with: pip install shap"
)


class SHAPEngine:
    """SHAP-based explainability for the XGBoost component model.

    Requires shap to be installed. If shap is unavailable, all public
    methods raise ImportError with a user-friendly install hint.
    """

    def __init__(self, xgb_model: XGBoostMatchModel) -> None:
        self._model = xgb_model
        self._explainer = None  # lazy-initialised on first call

    def _get_explainer(self):
        if not _SHAP_AVAILABLE:
            raise ImportError(_SHAP_UNAVAILABLE_MSG)
        if self._explainer is None:
            booster = self._model.pipeline["xgb"].get_booster()
            self._explainer = shap.TreeExplainer(booster)
            logger.info("SHAP TreeExplainer initialised.")
        return self._explainer

    def global_shap(self, X: pd.DataFrame) -> dict[str, float]:
        """Compute mean |SHAP value| per feature across all outcome classes.

        Returns dict mapping feature name → mean absolute SHAP value.
        """
        explainer = self._get_explainer()
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            arr = np.stack(shap_values, axis=-1)
        else:
            arr = shap_values

        if arr.ndim == 3:
            mean_abs = np.mean(np.abs(arr), axis=(0, 2))
        else:
            mean_abs = np.mean(np.abs(arr), axis=0)

        feature_names = getattr(self._model, 'feature_names', None) or FEATURE_COLUMNS
        return {
            name: float(round(val, 6))
            for name, val in zip(feature_names, mean_abs)
        }

    def local_shap(
        self,
        X: pd.DataFrame,
        outcome: str = "H",
    ) -> list[dict]:
        """Compute per-feature SHAP values for a single match.

        Returns list of dicts sorted by |shap_value| descending:
            [{"feature": "elo_diff", "shap_value": 0.18, "raw_value": 120.5}, ...]
        """
        explainer = self._get_explainer()
        shap_values = explainer.shap_values(X)

        classes = ["A", "D", "H"]
        class_idx = classes.index(outcome) if outcome in classes else 2

        if isinstance(shap_values, list):
            arr = shap_values[class_idx]
        else:
            arr = shap_values[:, :, class_idx] if shap_values.ndim == 3 else shap_values

        feature_names = getattr(self._model, 'feature_names', None) or FEATURE_COLUMNS
        row = arr[0] if arr.ndim == 2 else arr
        raw_values = X.iloc[0].to_dict()

        result = [
            {
                "feature": name,
                "shap_value": float(round(val, 6)),
                "raw_value": float(raw_values.get(name, 0.0)),
            }
            for name, val in zip(feature_names, row)
        ]
        result.sort(key=lambda d: abs(d["shap_value"]), reverse=True)
        return result
