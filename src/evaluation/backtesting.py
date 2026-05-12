"""Rolling chronological backtesting — train on past, evaluate on near future.

NEVER shuffles data. Always train on past, test on future.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta
from loguru import logger

from src.evaluation.metrics import compute_all_metrics, CLASSES
from src.features.feature_builder import FeatureBuilder, FEATURE_COLUMNS
from src.ml.xgboost_model import XGBoostMatchModel
from src.models.elo_model import EloModel
from src.models.poisson_model import PoissonModel


@dataclass
class BacktestResult:
    """Metrics for a single rolling backtest window."""

    window_start: str          # ISO date string
    window_end: str
    train_size: int
    test_size: int
    metrics: dict[str, float]  # output of compute_all_metrics


class RollingBacktester:
    """Evaluate prediction quality over time using rolling train/test windows.

    NEVER shuffles data. Always train on past, test on future.
    """

    def __init__(
        self,
        matches_df: pd.DataFrame,
        elo_model: EloModel,
        poisson_model: PoissonModel,
        min_train_years: int = 5,
        test_window_months: int = 6,
        step_months: int = 6,
    ) -> None:
        """
        Parameters
        ----------
        matches_df:
            Full match history. Must contain columns:
            date, home_team, away_team, home_goals, away_goals.
        elo_model:
            An EloModel (used for its config; a fresh instance is trained per window).
        poisson_model:
            A PoissonModel (refitted per window).
        min_train_years:
            Minimum years of history before the first test window starts.
        test_window_months:
            Width of each test window in months.
        step_months:
            Number of months to advance the window each iteration.
        """
        self._matches = matches_df.copy()
        self._matches["date"] = pd.to_datetime(self._matches["date"])
        self._elo_model = elo_model
        self._poisson_model = poisson_model
        self._min_train_years = min_train_years
        self._test_window_months = test_window_months
        self._step_months = step_months

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> list[BacktestResult]:
        """Execute rolling backtest.

        Returns list of BacktestResult (one per window).

        For each window:
        1. Train XGBoostMatchModel on all data up to window_start.
        2. Build features for test window matches.
        3. Compute compute_all_metrics on the test window.
        4. Append BacktestResult.
        5. Slide window forward by step_months.
        """
        sorted_matches = self._matches.sort_values("date").reset_index(drop=True)

        first_date: pd.Timestamp = sorted_matches["date"].min()
        last_date: pd.Timestamp = sorted_matches["date"].max()

        # First window_start = first_date + min_train_years
        window_start: pd.Timestamp = first_date + relativedelta(years=self._min_train_years)

        results: list[BacktestResult] = []

        while True:
            window_end: pd.Timestamp = window_start + relativedelta(months=self._test_window_months)

            # Stop when window_end exceeds available data
            if window_start >= last_date:
                break

            # Cap window_end at last available date
            effective_end = min(window_end, last_date)

            # Training set: all data strictly before window_start
            train_mask = sorted_matches["date"] < window_start
            train_df = sorted_matches[train_mask].reset_index(drop=True)

            # Test set: data in [window_start, effective_end)
            test_mask = (
                (sorted_matches["date"] >= window_start)
                & (sorted_matches["date"] < effective_end)
            )
            test_df = sorted_matches[test_mask].reset_index(drop=True)

            if len(test_df) < 10:
                logger.debug(
                    "Skipping window {}/{}: only {} test matches",
                    window_start.date(),
                    effective_end.date(),
                    len(test_df),
                )
                window_start += relativedelta(months=self._step_months)
                continue

            if len(train_df) == 0:
                logger.warning(
                    "No training data before {}; skipping window", window_start.date()
                )
                window_start += relativedelta(months=self._step_months)
                continue

            logger.info(
                "Backtest window {}/{}: {} train, {} test",
                window_start.date(),
                effective_end.date(),
                len(train_df),
                len(test_df),
            )

            # --- Train fresh Elo model (replay chronologically) ---
            fresh_elo = EloModel(config=self._elo_model.config)
            fresh_elo.train_on_matches(train_df)

            # --- Fit fresh Poisson model ---
            fresh_poisson = PoissonModel(config=self._poisson_model.config)
            fresh_poisson.fit(train_df)

            # --- Build XGBoost feature matrix using only training data ---
            try:
                feature_builder = FeatureBuilder(
                    matches_df=train_df,
                    elo_model=fresh_elo,
                    poisson_model=fresh_poisson,
                )
                train_feature_df = feature_builder.build_training_matrix()

                # Verify all 3 classes present in train labels
                if set(train_feature_df["label"].unique()) != {"A", "D", "H"}:
                    logger.warning(
                        "Window {}: training set missing outcome class(es); skipping",
                        window_start.date(),
                    )
                    window_start += relativedelta(months=self._step_months)
                    continue

                # Fit XGBoost on training features
                X_train = train_feature_df[FEATURE_COLUMNS]
                y_train = train_feature_df["label"]
                xgb_model = XGBoostMatchModel()
                xgb_model.fit(X_train, y_train)

                # --- Build features for the test window ---
                # Use a feature builder that has access only to data up to window_start
                test_feature_builder = FeatureBuilder(
                    matches_df=train_df,
                    elo_model=fresh_elo,
                    poisson_model=fresh_poisson,
                )

                test_rows = []
                for _, row in test_df.iterrows():
                    match_date = pd.Timestamp(row["date"])
                    feat = test_feature_builder.build_features_for_match(
                        home_team=str(row["home_team"]),
                        away_team=str(row["away_team"]),
                        match_date=match_date,
                    )
                    label = _result_label(int(row["home_goals"]), int(row["away_goals"]))
                    test_rows.append({**feat, "label": label})

                test_feature_df = pd.DataFrame(test_rows)

                if test_feature_df.empty:
                    window_start += relativedelta(months=self._step_months)
                    continue

                X_test = test_feature_df[FEATURE_COLUMNS]
                y_test = test_feature_df["label"]

                proba = xgb_model.predict_proba(X_test)
                metrics = compute_all_metrics(y_test, proba)

            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Window {}/{} failed: {}",
                    window_start.date(),
                    effective_end.date(),
                    exc,
                )
                window_start += relativedelta(months=self._step_months)
                continue

            logger.info(
                "Window {}/{} metrics: {}",
                window_start.date(),
                effective_end.date(),
                metrics,
            )

            results.append(
                BacktestResult(
                    window_start=window_start.date().isoformat(),
                    window_end=effective_end.date().isoformat(),
                    train_size=len(train_df),
                    test_size=len(test_df),
                    metrics=metrics,
                )
            )

            window_start += relativedelta(months=self._step_months)

        logger.info("Backtest complete: {} windows evaluated", len(results))
        return results

    def to_dataframe(self, results: list[BacktestResult]) -> pd.DataFrame:
        """Convert results list to DataFrame for saving/plotting."""
        rows = []
        for r in results:
            row = {
                "window_start": r.window_start,
                "window_end": r.window_end,
                "train_size": r.train_size,
                "test_size": r.test_size,
            }
            row.update(r.metrics)
            rows.append(row)
        return pd.DataFrame(rows)

    def save(self, results: list[BacktestResult], output_dir: Path) -> Path:
        """Save results to output_dir/backtest_results.csv. Returns path."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "backtest_results.csv"
        df = self.to_dataframe(results)
        df.to_csv(csv_path, index=False)
        logger.info("Backtest results saved → {}", csv_path)
        return csv_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result_label(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals == away_goals:
        return "D"
    return "A"
