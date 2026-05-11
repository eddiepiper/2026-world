"""Tests for src/ml/ — XGBoostMatchModel, ModelTrainer, MatchPredictor.

All tests use small synthetic DataFrames. The real 37k-row CSV is never loaded.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.features.feature_builder import FEATURE_COLUMNS
from src.ml.xgboost_model import XGBoostMatchModel
from src.ml.trainer import ModelTrainer
from src.ml.predictor import MatchPredictor
from src.models.elo_model import EloModel
from src.models.poisson_model import PoissonModel


# ---------------------------------------------------------------------------
# Shared synthetic data helpers
# ---------------------------------------------------------------------------

def _make_feature_df(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Build a minimal feature DataFrame with all FEATURE_COLUMNS + label + date."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        rng.random((n, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS,
    )
    # Assign labels roughly equally across 3 classes
    labels = (["H"] * (n // 3) + ["D"] * (n // 3) + ["A"] * (n - 2 * (n // 3)))
    rng.shuffle(labels)
    df["label"] = labels
    # Chronological dates spanning 2018-01-01 to 2022-01-01
    dates = pd.date_range("2018-01-01", periods=n, freq="7D")
    df["date"] = dates
    return df


def _make_matches_df(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Build a minimal matches DataFrame."""
    rng = np.random.default_rng(seed)
    teams = ["Alpha", "Beta", "Gamma", "Delta"]
    rows = []
    dates = pd.date_range("2018-01-01", periods=n, freq="7D")
    for i in range(n):
        h, a = rng.choice(teams, size=2, replace=False)
        hg = int(rng.integers(0, 4))
        ag = int(rng.integers(0, 4))
        rows.append({"date": dates[i], "home_team": h, "away_team": a,
                     "home_goals": hg, "away_goals": ag})
    return pd.DataFrame(rows)


@pytest.fixture()
def feature_df() -> pd.DataFrame:
    return _make_feature_df(n=90)


@pytest.fixture()
def X_y(feature_df: pd.DataFrame):
    X = feature_df[FEATURE_COLUMNS]
    y = feature_df["label"]
    return X, y


@pytest.fixture()
def fitted_model(X_y) -> XGBoostMatchModel:
    X, y = X_y
    model = XGBoostMatchModel(params={
        "n_estimators": 10,  # fast for tests
        "max_depth": 2,
        "learning_rate": 0.1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "random_state": 42,
        "eval_metric": "mlogloss",
    })
    model.fit(X, y)
    return model


@pytest.fixture()
def small_trained_elo(matches_df_small: pd.DataFrame) -> EloModel:
    model = EloModel()
    model.train_on_matches(matches_df_small)
    return model


@pytest.fixture()
def small_trained_poisson(matches_df_small: pd.DataFrame) -> PoissonModel:
    model = PoissonModel()
    model.fit(matches_df_small)
    return model


@pytest.fixture()
def matches_df_small() -> pd.DataFrame:
    return _make_matches_df(n=60)


# ---------------------------------------------------------------------------
# XGBoostMatchModel tests
# ---------------------------------------------------------------------------

class TestXGBoostModelFitAndPredictProba:
    def test_xgboost_model_fit_and_predict_proba(self, fitted_model, X_y):
        X, _ = X_y
        proba = fitted_model.predict_proba(X)
        assert proba.shape == (len(X), 3)

    def test_xgboost_model_predict_proba_sums_to_one(self, fitted_model, X_y):
        X, _ = X_y
        proba = fitted_model.predict_proba(X)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_xgboost_model_predict_proba_dict_keys(self, fitted_model, X_y):
        X, _ = X_y
        result = fitted_model.predict_proba_dict(X)
        assert len(result) == len(X)
        for row in result:
            assert set(row.keys()) == {"home_win", "draw", "away_win"}

    def test_xgboost_model_predict_proba_dict_sums_to_one(self, fitted_model, X_y):
        X, _ = X_y
        result = fitted_model.predict_proba_dict(X)
        for row in result:
            total = row["home_win"] + row["draw"] + row["away_win"]
            assert abs(total - 1.0) < 1e-6

    def test_xgboost_model_feature_names_set_after_fit(self, fitted_model):
        assert fitted_model.feature_names is not None
        assert fitted_model.feature_names == FEATURE_COLUMNS

    def test_xgboost_model_classes_sorted_alphabetically(self, fitted_model):
        assert fitted_model.classes_ == ["A", "D", "H"]


class TestXGBoostModelSaveLoad:
    def test_xgboost_model_save_and_load(self, fitted_model, X_y):
        X, _ = X_y
        original_proba = fitted_model.predict_proba(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "model.pkl"
            fitted_model.save(path)
            assert path.exists()

            loaded = XGBoostMatchModel.load(path)
            loaded_proba = loaded.predict_proba(X)

        np.testing.assert_allclose(original_proba, loaded_proba, atol=1e-6)

    def test_loaded_model_has_same_feature_names(self, fitted_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            fitted_model.save(path)
            loaded = XGBoostMatchModel.load(path)

        assert loaded.feature_names == fitted_model.feature_names

    def test_loaded_model_has_same_classes(self, fitted_model):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            fitted_model.save(path)
            loaded = XGBoostMatchModel.load(path)

        assert loaded.classes_ == fitted_model.classes_


# ---------------------------------------------------------------------------
# ModelTrainer tests
# ---------------------------------------------------------------------------

class TestModelTrainerSplit:
    """Use a pre-built feature_df; inject it via a patched build_feature_matrix."""

    def _make_trainer(self, matches_df_small, small_trained_elo, small_trained_poisson):
        return ModelTrainer(
            matches_df=matches_df_small,
            elo_model=small_trained_elo,
            poisson_model=small_trained_poisson,
            test_split_date="2020-01-01",
        )

    def test_trainer_split_chronological(self, matches_df_small, small_trained_elo, small_trained_poisson):
        trainer = self._make_trainer(matches_df_small, small_trained_elo, small_trained_poisson)
        feature_df = _make_feature_df(n=90)  # dates 2018..2021
        train_df, test_df = trainer.split(feature_df)
        # All train rows must be before 2020-01-01
        assert (train_df["date"] < pd.Timestamp("2020-01-01")).all()
        # All test rows must be on/after 2020-01-01
        assert (test_df["date"] >= pd.Timestamp("2020-01-01")).all()

    def test_trainer_split_no_overlap(self, matches_df_small, small_trained_elo, small_trained_poisson):
        trainer = self._make_trainer(matches_df_small, small_trained_elo, small_trained_poisson)
        feature_df = _make_feature_df(n=90)
        train_df, test_df = trainer.split(feature_df)
        # No date should appear in both sets
        train_dates = set(train_df["date"].dt.date)
        test_dates = set(test_df["date"].dt.date)
        assert train_dates.isdisjoint(test_dates)

    def test_trainer_split_covers_all_rows(self, matches_df_small, small_trained_elo, small_trained_poisson):
        trainer = self._make_trainer(matches_df_small, small_trained_elo, small_trained_poisson)
        feature_df = _make_feature_df(n=90)
        train_df, test_df = trainer.split(feature_df)
        assert len(train_df) + len(test_df) == len(feature_df)

    def test_trainer_train_returns_fitted_model(self, matches_df_small, small_trained_elo, small_trained_poisson):
        trainer = self._make_trainer(matches_df_small, small_trained_elo, small_trained_poisson)
        # Override build_feature_matrix to use synthetic data (avoids 37k-row computation)
        trainer._feature_df = _make_feature_df(n=90)
        model = trainer.train()
        assert isinstance(model, XGBoostMatchModel)
        assert model.pipeline is not None
        assert model.feature_names is not None

    def test_trainer_evaluate_returns_expected_keys(self, matches_df_small, small_trained_elo, small_trained_poisson):
        # Use a split date inside the synthetic date range (2018-01-01 + 90*7d ≈ 2019-09)
        trainer = ModelTrainer(
            matches_df=matches_df_small,
            elo_model=small_trained_elo,
            poisson_model=small_trained_poisson,
            test_split_date="2019-01-01",  # splits n=90 data (2018–2019) into train+test
        )
        feature_df = _make_feature_df(n=90)
        trainer._feature_df = feature_df
        train_df, test_df = trainer.split(feature_df)
        # Ensure test set is non-empty
        assert len(test_df) > 0, "Test set must be non-empty for evaluation"

        model = trainer.train()
        metrics = trainer.evaluate(model, test_df)

        assert set(metrics.keys()) == {"accuracy", "log_loss", "brier_score"}
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["log_loss"] >= 0.0
        assert metrics["brier_score"] >= 0.0

    def test_trainer_save_artifacts_creates_files(self, matches_df_small, small_trained_elo, small_trained_poisson):
        trainer = self._make_trainer(matches_df_small, small_trained_elo, small_trained_poisson)
        feature_df = _make_feature_df(n=90)
        trainer._feature_df = feature_df
        model = trainer.train()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "model_output"
            metadata = {"train_size": 60, "test_size": 30, "split_date": "2020-01-01",
                        "test_metrics": {}, "feature_count": len(FEATURE_COLUMNS), "timestamp": "2026-01-01"}
            trainer.save_artifacts(model, output_dir, metadata)

            assert (output_dir / "model.pkl").exists()
            assert (output_dir / "feature_names.json").exists()
            assert (output_dir / "metadata.json").exists()

            with open(output_dir / "feature_names.json") as f:
                names = json.load(f)
            assert names == FEATURE_COLUMNS

            with open(output_dir / "metadata.json") as f:
                saved_meta = json.load(f)
            assert saved_meta["train_size"] == 60


# ---------------------------------------------------------------------------
# MatchPredictor tests
# ---------------------------------------------------------------------------

class TestMatchPredictor:
    """Use a fitted model + a mock FeatureBuilder to avoid full pipeline."""

    @pytest.fixture()
    def mock_feature_builder(self):
        """FeatureBuilder that returns a fixed feature dict."""
        rng = np.random.default_rng(1)
        fixed_features = dict(zip(FEATURE_COLUMNS, rng.random(len(FEATURE_COLUMNS)).tolist()))
        fb = MagicMock()
        fb.build_features_for_match.return_value = fixed_features
        return fb

    @pytest.fixture()
    def predictor(self, fitted_model, mock_feature_builder):
        return MatchPredictor(model=fitted_model, feature_builder=mock_feature_builder)

    def test_predictor_predict_returns_keys(self, predictor):
        result = predictor.predict("Brazil", "France")
        assert set(result.keys()) == {"home_win", "draw", "away_win"}

    def test_predictor_predict_sums_to_one(self, predictor):
        result = predictor.predict("Brazil", "France")
        total = result["home_win"] + result["draw"] + result["away_win"]
        assert abs(total - 1.0) < 1e-6

    def test_predictor_predict_all_non_negative(self, predictor):
        result = predictor.predict("Brazil", "France")
        assert result["home_win"] >= 0.0
        assert result["draw"] >= 0.0
        assert result["away_win"] >= 0.0

    def test_predictor_predict_uses_today_when_date_none(self, predictor, mock_feature_builder):
        predictor.predict("Brazil", "France", match_date=None)
        call_args = mock_feature_builder.build_features_for_match.call_args
        match_date_used = call_args.kwargs.get("match_date") or call_args.args[2]
        assert isinstance(match_date_used, pd.Timestamp)

    def test_predictor_from_artifacts(self, fitted_model, mock_feature_builder):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            fitted_model.save(output_dir / "model.pkl")

            predictor = MatchPredictor.from_artifacts(output_dir, mock_feature_builder)
            result = predictor.predict("Spain", "Germany")

        assert set(result.keys()) == {"home_win", "draw", "away_win"}
        total = result["home_win"] + result["draw"] + result["away_win"]
        assert abs(total - 1.0) < 1e-6
