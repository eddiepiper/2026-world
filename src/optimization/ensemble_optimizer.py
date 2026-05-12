from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize

from src.config.settings import settings
from src.evaluation.metrics import CLASSES, log_loss_score
from src.features.feature_builder import FEATURE_COLUMNS, FeatureBuilder
from src.models.elo_model import EloModel
from src.models.poisson_model import PoissonModel
from src.ml.xgboost_model import XGBoostMatchModel
from src.optimization.weight_search import validate_weights


class WeightOptimizer:
    def __init__(
        self,
        matches_df: pd.DataFrame,
        elo_model: EloModel,
        poisson_model: PoissonModel,
        xgb_model: XGBoostMatchModel,
        feature_builder: FeatureBuilder,
        test_split_date: str | None = None,
    ) -> None:
        self._matches = matches_df
        self._elo = elo_model
        self._poisson = poisson_model
        self._xgb = xgb_model
        self._feature_builder = feature_builder
        self._split_date = test_split_date or settings.ml.test_split_date

    def optimize(self) -> dict:
        """Find weights minimizing log loss on the test split using SLSQP."""
        elo_probs, poisson_probs, xgb_probs, y_true = self._build_test_arrays()

        if len(y_true) == 0:
            raise ValueError("No test data available after the split date.")

        baseline_weights = np.array([0.3, 0.3, 0.4])
        baseline_ll = self._log_loss_for_weights(
            baseline_weights, elo_probs, poisson_probs, xgb_probs, y_true
        )

        def objective(w: np.ndarray) -> float:
            return self._log_loss_for_weights(w, elo_probs, poisson_probs, xgb_probs, y_true)

        result = minimize(
            objective,
            x0=baseline_weights,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * 3,
            constraints=[{"type": "eq", "fun": lambda w: float(w.sum()) - 1.0}],
        )

        if not result.success:
            logger.warning(f"Optimizer did not converge: {result.message}")

        w = result.x
        validate_weights(float(w[0]), float(w[1]), float(w[2]))

        optimized_ll = float(result.fun)
        improvement = baseline_ll - optimized_ll

        output = {
            "elo": round(float(w[0]), 6),
            "poisson": round(float(w[1]), 6),
            "xgboost": round(float(w[2]), 6),
            "optimized_log_loss": round(optimized_ll, 6),
            "baseline_log_loss": round(baseline_ll, 6),
            "improvement": round(improvement, 6),
            "method": "SLSQP",
            "validated_at": str(date.today()),
        }
        logger.info(
            f"Optimized weights: elo={output['elo']:.3f}, "
            f"poisson={output['poisson']:.3f}, xgboost={output['xgboost']:.3f}, "
            f"improvement={improvement:.4f}"
        )
        return output

    def save(self, path: Path, result: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
        logger.info(f"Weight config saved to {path}")

    def _build_test_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Build probability arrays for the test split. Column order: [A, D, H]."""
        test_df = self._matches[self._matches["date"] >= self._split_date].copy()

        elo_list: list[list[float]] = []
        poisson_list: list[list[float]] = []
        xgb_list: list[list[float]] = []
        y_true: list[str] = []

        for _, row in test_df.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            outcome = row["result"]  # "H", "D", or "A"
            match_date = pd.Timestamp(row["date"])

            # Elo: returns (win, draw, loss) = (H, D, A) → reorder to [A, D, H]
            ew, ed, el = self._elo.win_draw_loss_probabilities(home, away)
            elo_list.append([el, ed, ew])

            # Poisson: same order (win, draw, loss) = (H, D, A) → reorder to [A, D, H]
            pw, pd_, pl = self._poisson.win_draw_loss_from_poisson(home, away)
            poisson_list.append([pl, pd_, pw])

            # XGBoost: build features DataFrame, then predict_proba_dict → [A, D, H]
            try:
                feature_dict = self._feature_builder.build_features_for_match(
                    home, away, match_date
                )
                # build_features_for_match returns a dict; wrap in DataFrame for XGBoost
                if isinstance(feature_dict, dict):
                    features_df = pd.DataFrame([feature_dict])
                    # Ensure only model feature columns are present, in order
                    cols = [c for c in FEATURE_COLUMNS if c in features_df.columns]
                    features_df = features_df[cols]
                else:
                    # Already a DataFrame
                    features_df = feature_dict

                xgb_pred = self._xgb.predict_proba_dict(features_df)
                # predict_proba_dict returns a list of dicts; take the first (single match)
                p = xgb_pred[0] if isinstance(xgb_pred, list) else xgb_pred
                xgb_list.append([p["away_win"], p["draw"], p["home_win"]])
                y_true.append(outcome)
            except Exception as exc:
                logger.debug(f"Skipping {home} vs {away}: {exc}")
                elo_list.pop()
                poisson_list.pop()

        return (
            np.array(elo_list),
            np.array(poisson_list),
            np.array(xgb_list),
            y_true,
        )

    @staticmethod
    def _log_loss_for_weights(
        w: np.ndarray,
        elo_probs: np.ndarray,
        poisson_probs: np.ndarray,
        xgb_probs: np.ndarray,
        y_true: list[str],
    ) -> float:
        blended = w[0] * elo_probs + w[1] * poisson_probs + w[2] * xgb_probs
        # Normalize rows
        row_sums = blended.sum(axis=1, keepdims=True)
        blended = blended / np.where(row_sums > 0, row_sums, 1.0)
        # log_loss_score accepts (y_true, proba) — labels=CLASSES is applied internally
        return log_loss_score(pd.Series(y_true), blended)
