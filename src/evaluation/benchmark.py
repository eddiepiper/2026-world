"""Model benchmarking — compare Elo, Poisson, XGBoost, and Ensemble on the same test set.

All models are evaluated on matches AFTER test_split_date only to ensure
a fair chronological comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.evaluation.metrics import CLASSES, compute_all_metrics
from src.features.feature_builder import FeatureBuilder, FEATURE_COLUMNS
from src.ml.xgboost_model import XGBoostMatchModel
from src.models.elo_model import EloModel
from src.models.poisson_model import PoissonModel


@dataclass
class ModelBenchmarkResult:
    """Benchmark metrics for a single model."""

    model_name: str
    metrics: dict[str, float]


class ModelBenchmarker:
    """Compare multiple prediction approaches on the same chronological test set.

    Evaluates four models:
    - Elo
    - Poisson
    - XGBoost
    - Ensemble (0.3 Elo / 0.3 Poisson / 0.4 XGBoost)

    All models are evaluated on matches AFTER test_split_date.
    """

    # Ensemble blend weights (Elo / Poisson / XGBoost)
    _ENSEMBLE_WEIGHTS = (0.3, 0.3, 0.4)
    _ENSEMBLE_NAME = "Ensemble (0.3/0.3/0.4)"

    def __init__(
        self,
        matches_df: pd.DataFrame,
        elo_model: EloModel,
        poisson_model: PoissonModel,
        xgb_model: XGBoostMatchModel,
        feature_builder: FeatureBuilder,
        test_split_date: str = "2020-01-01",
    ) -> None:
        """
        Parameters
        ----------
        matches_df:
            Full match history with columns:
            date, home_team, away_team, home_goals, away_goals.
        elo_model:
            A trained EloModel.
        poisson_model:
            A trained PoissonModel.
        xgb_model:
            A fitted XGBoostMatchModel.
        feature_builder:
            A FeatureBuilder for XGBoost inference features.
        test_split_date:
            ISO date string; only matches on/after this date are in the test set.
        """
        self._matches = matches_df.copy()
        self._matches["date"] = pd.to_datetime(self._matches["date"])
        self._elo = elo_model
        self._poisson = poisson_model
        self._xgb = xgb_model
        self._feature_builder = feature_builder
        self._test_split_date = test_split_date

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> list[ModelBenchmarkResult]:
        """Evaluate all 4 models on the test set.

        Returns list of ModelBenchmarkResult for:
        - "Elo"
        - "Poisson"
        - "XGBoost"
        - "Ensemble (0.3/0.3/0.4)"

        All evaluated on matches AFTER test_split_date only.
        """
        split_ts = pd.Timestamp(self._test_split_date)
        test_df = self._matches[self._matches["date"] >= split_ts].reset_index(drop=True)

        if test_df.empty:
            logger.warning(
                "No test matches found on/after {}. Returning empty benchmark.",
                self._test_split_date,
            )
            return []

        logger.info(
            "Benchmarking on {} test matches (split date: {})",
            len(test_df),
            self._test_split_date,
        )

        # Build true labels
        y_true = pd.Series(
            [
                _result_label(int(r["home_goals"]), int(r["away_goals"]))
                for _, r in test_df.iterrows()
            ]
        )

        # --- 1. Elo probabilities ---
        elo_proba = self._elo_proba(test_df)
        logger.info("Elo probabilities computed.")

        # --- 2. Poisson probabilities ---
        poisson_proba = self._poisson_proba(test_df)
        logger.info("Poisson probabilities computed.")

        # --- 3. XGBoost probabilities ---
        xgb_proba = self._xgb_proba(test_df)
        logger.info("XGBoost probabilities computed.")

        # --- 4. Ensemble probabilities ---
        w_elo, w_poi, w_xgb = self._ENSEMBLE_WEIGHTS
        ensemble_proba = (
            w_elo * elo_proba
            + w_poi * poisson_proba
            + w_xgb * xgb_proba
        )
        # Normalise row-wise
        row_sums = ensemble_proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)  # avoid div-by-zero
        ensemble_proba = ensemble_proba / row_sums
        logger.info("Ensemble probabilities computed.")

        results = [
            ModelBenchmarkResult(
                model_name="Elo",
                metrics=compute_all_metrics(y_true, elo_proba),
            ),
            ModelBenchmarkResult(
                model_name="Poisson",
                metrics=compute_all_metrics(y_true, poisson_proba),
            ),
            ModelBenchmarkResult(
                model_name="XGBoost",
                metrics=compute_all_metrics(y_true, xgb_proba),
            ),
            ModelBenchmarkResult(
                model_name=self._ENSEMBLE_NAME,
                metrics=compute_all_metrics(y_true, ensemble_proba),
            ),
        ]

        for r in results:
            logger.info("  {} → {}", r.model_name, r.metrics)

        return results

    def to_dataframe(self, results: list[ModelBenchmarkResult]) -> pd.DataFrame:
        """Convert to wide DataFrame: rows = models, columns = metrics."""
        rows = []
        for r in results:
            row = {"model": r.model_name}
            row.update(r.metrics)
            rows.append(row)
        return pd.DataFrame(rows).set_index("model")

    def save(
        self,
        results: list[ModelBenchmarkResult],
        output_dir: Path,
    ) -> tuple[Path, Path]:
        """Save CSV and markdown summary.

        Returns (csv_path, md_path).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe(results)

        # CSV
        csv_path = output_dir / "benchmark_results.csv"
        df.to_csv(csv_path)
        logger.info("Benchmark CSV saved → {}", csv_path)

        # Markdown table (manual construction to avoid tabulate dependency)
        md_path = output_dir / "benchmark_summary.md"
        md_lines = ["# Model Benchmark Summary", ""]
        # Header row
        metric_cols = list(df.columns)
        header = "| Model | " + " | ".join(metric_cols) + " |"
        separator = "| --- | " + " | ".join(["---"] * len(metric_cols)) + " |"
        md_lines.append(header)
        md_lines.append(separator)
        for model_name, row in df.iterrows():
            values = " | ".join(f"{v:.4f}" for v in row)
            md_lines.append(f"| {model_name} | {values} |")
        md_lines.append("")
        md_path.write_text("\n".join(md_lines))
        logger.info("Benchmark markdown saved → {}", md_path)

        return csv_path, md_path

    def best_model(self, results: list[ModelBenchmarkResult]) -> str:
        """Return the name of the model with lowest log_loss."""
        if not results:
            raise ValueError("results list is empty")
        return min(results, key=lambda r: r.metrics["log_loss"]).model_name

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _elo_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        """Convert Elo win/draw/loss to (n, 3) array ordered [A, D, H]."""
        rows = []
        for _, row in test_df.iterrows():
            p_win, p_draw, p_loss = self._elo.win_draw_loss_probabilities(
                str(row["home_team"]), str(row["away_team"])
            )
            # Elo returns (p_home_win, p_draw, p_away_win)
            # CLASSES order: A=0, D=1, H=2
            rows.append([p_loss, p_draw, p_win])  # [A, D, H]
        return np.array(rows, dtype=float)

    def _poisson_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        """Convert Poisson win/draw/loss to (n, 3) array ordered [A, D, H]."""
        rows = []
        for _, row in test_df.iterrows():
            p_win, p_draw, p_loss = self._poisson.win_draw_loss_from_poisson(
                str(row["home_team"]), str(row["away_team"])
            )
            # Poisson returns (p_home_win, p_draw, p_away_win)
            # CLASSES order: A=0, D=1, H=2
            rows.append([p_loss, p_draw, p_win])  # [A, D, H]
        return np.array(rows, dtype=float)

    def _xgb_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        """Build XGBoost features and return (n, 3) proba array [A, D, H]."""
        feature_rows = []
        for _, row in test_df.iterrows():
            match_date = pd.Timestamp(row["date"])
            feat = self._feature_builder.build_features_for_match(
                home_team=str(row["home_team"]),
                away_team=str(row["away_team"]),
                match_date=match_date,
            )
            feature_rows.append(feat)

        X = pd.DataFrame(feature_rows)[FEATURE_COLUMNS]
        return self._xgb.predict_proba(X)  # already [A, D, H]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result_label(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals == away_goals:
        return "D"
    return "A"
