"""Training orchestration for the XGBoost match prediction model."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, log_loss

from src.config.settings import settings
from src.features.feature_builder import FeatureBuilder, FEATURE_COLUMNS
from src.ml.calibration import ProbabilityCalibrator
from src.ml.xgboost_model import XGBoostMatchModel
from src.models.elo_model import EloModel
from src.models.poisson_model import PoissonModel


class ModelTrainer:
    """Orchestrates feature building, train/test split, fitting, evaluation, and artifact saving."""

    def __init__(
        self,
        matches_df: pd.DataFrame,
        elo_model: EloModel,
        poisson_model: PoissonModel,
        rankings_df: pd.DataFrame | None = None,
        test_split_date: str | None = None,
        random_seed: int | None = None,
    ) -> None:
        self._matches_df = matches_df
        self._elo_model = elo_model
        self._poisson_model = poisson_model
        self._rankings_df = rankings_df
        self._test_split_date = test_split_date or settings.ml.test_split_date
        self._random_seed = random_seed or settings.ml.random_seed

        self._feature_builder = FeatureBuilder(
            matches_df=matches_df,
            elo_model=elo_model,
            poisson_model=poisson_model,
            rankings_df=rankings_df,
        )

        # Cache for the full feature matrix (built lazily)
        self._feature_df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_feature_matrix(self) -> pd.DataFrame:
        """Use FeatureBuilder to build full feature+label matrix. Cached after first call.

        The returned DataFrame also includes a ``date`` column copied from the
        original matches for use in chronological splitting.
        """
        if self._feature_df is not None:
            return self._feature_df

        logger.info("Building feature matrix from {} matches…", len(self._matches_df))
        feature_df = self._feature_builder.build_training_matrix()

        # Attach the date column so split() can filter by date
        sorted_matches = self._matches_df.copy()
        sorted_matches["date"] = pd.to_datetime(sorted_matches["date"])
        sorted_matches = sorted_matches.sort_values("date").reset_index(drop=True)
        feature_df = feature_df.copy()
        feature_df["date"] = sorted_matches["date"].values

        self._feature_df = feature_df
        logger.info("Feature matrix ready: {} rows × {} feature cols", len(feature_df), len(FEATURE_COLUMNS))
        return self._feature_df

    def split(self, feature_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split by date: rows before test_split_date → train, on/after → test.

        Returns (train_df, test_df). Both include all feature cols + 'label'.
        The ``date`` column is retained in both splits.
        """
        split_ts = pd.Timestamp(self._test_split_date)
        train_mask = feature_df["date"] < split_ts
        train_df = feature_df[train_mask].reset_index(drop=True)
        test_df = feature_df[~train_mask].reset_index(drop=True)
        logger.info(
            "Train/test split at {}: {} train rows, {} test rows",
            self._test_split_date,
            len(train_df),
            len(test_df),
        )
        return train_df, test_df

    def train(self) -> XGBoostMatchModel:
        """Build features, split, fit model on train set, return fitted model."""
        feature_df = self.build_feature_matrix()
        train_df, _ = self.split(feature_df)

        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df["label"]

        logger.info("Fitting XGBoostMatchModel on {} training samples…", len(X_train))
        params = {**XGBoostMatchModel().params, "random_state": self._random_seed}
        model = XGBoostMatchModel(params=params)
        model.fit(X_train, y_train)
        logger.info("Model training complete.")
        return model

    def evaluate(self, model: XGBoostMatchModel, test_df: pd.DataFrame) -> dict[str, float]:
        """Evaluate model on test_df.

        Returns dict with: accuracy, log_loss, brier_score.
        """
        if test_df.empty:
            logger.warning("evaluate() called with empty test_df — returning NaN metrics")
            return {"accuracy": float("nan"), "log_loss": float("nan"), "brier_score": float("nan")}

        X_test = test_df[FEATURE_COLUMNS]
        y_test = test_df["label"]

        proba = model.predict_proba(X_test)  # shape (n, 3): A, D, H columns
        classes = model.classes_  # ['A', 'D', 'H']

        # Hard predictions (argmax → class index → label string)
        pred_indices = np.argmax(proba, axis=1)
        y_pred = [classes[i] for i in pred_indices]

        acc = float(accuracy_score(y_test, y_pred))
        ll = float(log_loss(y_test, proba, labels=classes))

        # Brier score (multiclass): mean squared error between one-hot targets and probabilities
        label_to_idx = {label: i for i, label in enumerate(classes)}
        one_hot = np.zeros_like(proba)
        for i, label in enumerate(y_test):
            one_hot[i, label_to_idx[label]] = 1.0
        brier = float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))

        metrics = {"accuracy": acc, "log_loss": ll, "brier_score": brier}
        logger.info("Evaluation metrics: {}", metrics)
        return metrics

    def save_artifacts(
        self,
        model: XGBoostMatchModel,
        output_dir: Path,
        metadata: dict | None = None,
    ) -> None:
        """Save model, feature_names.json, and metadata.json to output_dir."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Model pickle
        model_path = output_dir / "model.pkl"
        model.save(model_path)
        logger.info("Model saved → {}", model_path)

        # 2. Feature names
        feature_names_path = output_dir / "feature_names.json"
        with open(feature_names_path, "w") as f:
            json.dump(model.feature_names or FEATURE_COLUMNS, f, indent=2)
        logger.info("Feature names saved → {}", feature_names_path)

        # 3. Metadata
        meta_path = output_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata or {}, f, indent=2)
        logger.info("Metadata saved → {}", meta_path)

    def run(self, output_dir: Path) -> dict[str, float]:
        """Full pipeline: build → split → train → evaluate → save. Returns metrics dict."""
        output_dir = Path(output_dir)

        # Build & split
        feature_df = self.build_feature_matrix()
        train_df, test_df = self.split(feature_df)

        # Train (reuse train() — no duplication)
        model = self.train()

        # Evaluate
        metrics = self.evaluate(model, test_df)

        # Save
        metadata = {
            "train_size": len(train_df),
            "test_size": len(test_df),
            "split_date": self._test_split_date,
            "test_metrics": metrics,
            "feature_count": len(FEATURE_COLUMNS),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.save_artifacts(model, output_dir, metadata)

        return metrics

    def calibrate(
        self,
        model: XGBoostMatchModel,
        cal_df: pd.DataFrame,
    ) -> ProbabilityCalibrator:
        """Fit and return a ProbabilityCalibrator using a calibration DataFrame.

        The caller is responsible for ensuring ``cal_df`` is a *held-out* split
        that was not used during model training (no leakage).

        Parameters
        ----------
        model:
            A fitted XGBoostMatchModel.
        cal_df:
            Calibration DataFrame containing all FEATURE_COLUMNS plus a
            ``'label'`` column with true outcome labels ('H', 'D', 'A').

        Returns
        -------
        ProbabilityCalibrator
            A fitted calibrator wrapping ``model``.
        """
        X_cal = cal_df[FEATURE_COLUMNS]
        y_cal = cal_df["label"]

        logger.info(
            "Fitting ProbabilityCalibrator on {} calibration samples…", len(X_cal)
        )
        calibrator = ProbabilityCalibrator(model)
        calibrator.fit(X_cal, y_cal)
        logger.info("Calibration complete.")
        return calibrator
