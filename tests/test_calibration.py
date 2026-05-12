"""Tests for src/ml/calibration.py."""
from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.feature_builder import FEATURE_COLUMNS
from src.ml.calibration import ProbabilityCalibrator
from src.ml.xgboost_model import XGBoostMatchModel


def _make_feature_df(n: int = 90, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({col: rng.uniform(0, 1, n) for col in FEATURE_COLUMNS})


def _make_labels(n: int = 90, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    labels = ["H"] * (n // 3) + ["D"] * (n // 3) + ["A"] * (n - 2 * (n // 3))
    rng.shuffle(labels)
    return pd.Series(labels)


def _compute_ece(proba: np.ndarray, y_true: pd.Series, classes: list[str], n_bins: int = 10) -> float:
    n_samples = len(y_true)
    ece_total = 0.0
    for idx, cls in enumerate(classes):
        binary = (y_true == cls).astype(float).values
        scores = proba[:, idx]
        edges = np.linspace(0, 1, n_bins + 1)
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (scores >= lo) & (scores < hi)
            if not mask.any():
                continue
            ece_total += mask.sum() / n_samples * abs(scores[mask].mean() - binary[mask].mean())
    return ece_total / len(classes)


@pytest.fixture(scope="module")
def trained_model() -> XGBoostMatchModel:
    X = _make_feature_df(n=120, seed=1)
    y = _make_labels(n=120, seed=1)
    m = XGBoostMatchModel()
    m.fit(X, y)
    return m


@pytest.fixture(scope="module")
def cal_X() -> pd.DataFrame:
    return _make_feature_df(n=40, seed=2)


@pytest.fixture(scope="module")
def cal_y() -> pd.Series:
    return _make_labels(n=40, seed=2)


@pytest.fixture(scope="module")
def fitted_calibrator(trained_model, cal_X, cal_y) -> ProbabilityCalibrator:
    cal = ProbabilityCalibrator(trained_model)
    cal.fit(cal_X, cal_y)
    return cal


class TestCalibratorFitAndPredict:
    def test_calibrator_fit_and_predict_proba(self, fitted_calibrator, cal_X):
        proba = fitted_calibrator.predict_proba(cal_X)
        assert proba.shape == (len(cal_X), 3)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_calibrator_predict_proba_sums_to_one(self, fitted_calibrator, cal_X):
        np.testing.assert_allclose(fitted_calibrator.predict_proba(cal_X).sum(axis=1), 1.0, atol=1e-6)

    def test_calibrator_predict_proba_dict_keys(self, fitted_calibrator, cal_X):
        for row in fitted_calibrator.predict_proba_dict(cal_X):
            assert set(row.keys()) == {"home_win", "draw", "away_win"}
            assert abs(sum(row.values()) - 1.0) < 1e-6

    def test_calibrator_classes_delegates_to_model(self, fitted_calibrator):
        assert fitted_calibrator.classes_ == ["A", "D", "H"]


class TestCalibratorRaisesIfNotFitted:
    def test_calibrator_raises_if_not_fitted(self, trained_model, cal_X):
        with pytest.raises(RuntimeError, match="not been fitted"):
            ProbabilityCalibrator(trained_model).predict_proba(cal_X)


class TestCalibratorPersistence:
    def test_calibrator_save_and_load_roundtrip(self, fitted_calibrator, cal_X):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cal.pkl"
            fitted_calibrator.save(p)
            loaded = ProbabilityCalibrator.load(p)
        np.testing.assert_allclose(
            fitted_calibrator.predict_proba(cal_X),
            loaded.predict_proba(cal_X),
            atol=1e-6,
        )

    def test_calibrator_load_wrong_type_raises(self, trained_model):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.pkl"
            with open(p, "wb") as f:
                pickle.dump(trained_model, f)
            with pytest.raises(TypeError):
                ProbabilityCalibrator.load(p)

    def test_calibrator_save_creates_parent_dirs(self, fitted_calibrator):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a" / "b" / "cal.pkl"
            fitted_calibrator.save(p)
            assert p.exists()


class TestCalibratorCalibrationQuality:
    def test_calibrator_improves_or_maintains_calibration(self, trained_model, cal_X, cal_y):
        X_eval = _make_feature_df(n=60, seed=99)
        y_eval = _make_labels(n=60, seed=99)
        raw_ece = _compute_ece(trained_model.predict_proba(X_eval), y_eval, trained_model.classes_)
        cal = ProbabilityCalibrator(trained_model)
        cal.fit(cal_X, cal_y)
        cal_ece = _compute_ece(cal.predict_proba(X_eval), y_eval, trained_model.classes_)
        assert cal_ece <= raw_ece * 1.5 + 0.05, f"Calibrated ECE ({cal_ece:.4f}) much worse than raw ({raw_ece:.4f})"


class TestTrainerCalibrateMethod:
    def test_trainer_calibrate_returns_calibrator(self):
        from src.ml.trainer import ModelTrainer
        from src.models.elo_model import EloModel
        from src.models.poisson_model import PoissonModel

        dates = pd.date_range("2010-01-01", periods=90, freq="7D")
        matches_df = pd.DataFrame({
            "date": dates,
            "home_team": ["Brazil", "France", "Germany"] * 30,
            "away_team": ["France", "Germany", "Brazil"] * 30,
            "home_goals": [2, 1, 0] * 30,
            "away_goals": [1, 1, 2] * 30,
            "tournament": ["Friendly"] * 90,
        })
        elo = EloModel()
        elo.train_on_matches(matches_df)
        poisson = PoissonModel()
        poisson.fit(matches_df)

        trainer = ModelTrainer(
            matches_df=matches_df,
            elo_model=elo,
            poisson_model=poisson,
            test_split_date="2011-01-01",
        )
        feature_df = trainer.build_feature_matrix()
        train_df, cal_df = trainer.split(feature_df)
        if cal_df.empty or len(train_df["label"].unique()) < 3:
            pytest.skip("Insufficient synthetic data for calibration test")
        model = trainer.train()
        calibrator = trainer.calibrate(model, cal_df)
        assert isinstance(calibrator, ProbabilityCalibrator)
        assert calibrator._is_fitted
