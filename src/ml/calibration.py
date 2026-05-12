"""Isotonic regression probability calibration for XGBoostMatchModel.

Wraps a trained XGBoostMatchModel and applies per-class isotonic regression
to improve probability reliability (calibration). Uses a held-out calibration
set — NOT the training set — to avoid data leakage.

Note: Persistence uses pickle. Only load calibrators from trusted sources.
"""
from __future__ import annotations

import pickle  # noqa: S403 — required for artifact persistence; only load from trusted sources
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.ml.xgboost_model import XGBoostMatchModel
from src.utils.helpers import normalize_probabilities


class ProbabilityCalibrator:
    """Wraps a trained XGBoostMatchModel and applies per-class isotonic calibration.

    One IsotonicRegression is fitted per outcome class (A, D, H).
    The calibrators must be fitted on a *held-out* calibration set, not on the
    training set, to prevent data leakage.

    Typical usage::

        calibrator = ProbabilityCalibrator(model)
        calibrator.fit(X_cal, y_cal)
        proba = calibrator.predict_proba(X_new)
    """

    def __init__(self, model: XGBoostMatchModel) -> None:
        """Wrap an already-trained XGBoostMatchModel.

        Parameters
        ----------
        model:
            A fitted XGBoostMatchModel instance.
        """
        self._model = model
        self._calibrators: dict[str, IsotonicRegression] = {}  # class label -> calibrator
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X_cal: pd.DataFrame, y_cal: pd.Series) -> None:
        """Fit one isotonic regressor per class on the calibration set.

        For each class C:
          - Binary targets = (y_cal == C).astype(int)
          - Predicted scores = model.predict_proba(X_cal)[:, class_index]
          - Fit IsotonicRegression(out_of_bounds="clip") on (scores, binary_targets)

        Parameters
        ----------
        X_cal:
            Feature DataFrame -- the *held-out* calibration split (not training data).
        y_cal:
            True labels ('H', 'D', 'A') for the calibration rows.
        """
        raw_proba = self._model.predict_proba(X_cal)  # shape (n, 3): columns A, D, H

        self._calibrators = {}
        for idx, class_label in enumerate(self._model.classes_):
            binary_targets = (y_cal == class_label).astype(int).values
            predicted_scores = raw_proba[:, idx]

            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(predicted_scores, binary_targets)
            self._calibrators[class_label] = ir

        self._is_fitted = True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated probability array of shape (n_samples, 3).

        Columns are ordered A, D, H -- identical to the wrapped model's
        ``classes_`` ordering.

        Steps:
        1. Get raw probabilities from the wrapped model.
        2. Apply each class's isotonic calibrator.
        3. Clip results to [0, 1].
        4. Normalise each row to sum to 1.

        Parameters
        ----------
        X:
            Feature DataFrame.

        Returns
        -------
        np.ndarray
            Shape (n_samples, 3) with columns ordered A, D, H.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "ProbabilityCalibrator has not been fitted. Call fit() first."
            )

        raw_proba = self._model.predict_proba(X)  # (n, 3)
        calibrated = np.empty_like(raw_proba)

        for idx, class_label in enumerate(self._model.classes_):
            scores = raw_proba[:, idx]
            calibrated_scores = self._calibrators[class_label].predict(scores)
            calibrated[:, idx] = np.clip(calibrated_scores, 0.0, 1.0)

        # Normalise each row to sum to 1
        normalised = np.array(
            [normalize_probabilities(list(row)) for row in calibrated],
            dtype=float,
        )
        return normalised

    def predict_proba_dict(self, X: pd.DataFrame) -> list[dict[str, float]]:
        """Return calibrated probabilities as a list of dicts.

        Each dict has keys: ``'home_win'``, ``'draw'``, ``'away_win'``.

        Parameters
        ----------
        X:
            Feature DataFrame.
        """
        proba = self.predict_proba(X)  # (n, 3): columns A, D, H

        classes = self._model.classes_
        idx_h = classes.index("H")
        idx_d = classes.index("D")
        idx_a = classes.index("A")

        results: list[dict[str, float]] = []
        for row in proba:
            results.append({
                "home_win": float(row[idx_h]),
                "draw": float(row[idx_d]),
                "away_win": float(row[idx_a]),
            })
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save the calibrator (including the wrapped model) to a pickle file.

        Creates parent directories if needed.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "ProbabilityCalibrator":
        """Load a saved ProbabilityCalibrator from a pickle file.

        Parameters
        ----------
        path:
            Path to the pickle file produced by :meth:`save`.

        Raises
        ------
        TypeError
            If the loaded object is not a :class:`ProbabilityCalibrator`.
        """
        with open(Path(path), "rb") as f:
            obj = pickle.load(f)  # noqa: S301
        if not isinstance(obj, cls):
            raise TypeError(f"Expected ProbabilityCalibrator, got {type(obj)}")
        return obj

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def classes_(self) -> list[str]:
        """Delegate to the wrapped model's ``classes_``."""
        return self._model.classes_
