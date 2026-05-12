"""Tests for src/evaluation/ — metrics, rolling backtest, model benchmarker.

All tests use small synthetic DataFrames. The real CSV is never loaded.
"""
from __future__ import annotations

import tempfile
from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    CLASSES,
    accuracy,
    brier_score,
    compute_all_metrics,
    expected_calibration_error,
    log_loss_score,
    ranked_probability_score,
)
from src.evaluation.backtesting import BacktestResult, RollingBacktester
from src.evaluation.benchmark import ModelBenchmarkResult, ModelBenchmarker
from src.features.feature_builder import FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _perfect_proba(labels: list[str]) -> np.ndarray:
    """Return a (n, 3) proba array where each row is 1.0 for the true class."""
    class_idx = {c: i for i, c in enumerate(CLASSES)}
    n = len(labels)
    proba = np.zeros((n, 3))
    for i, label in enumerate(labels):
        proba[i, class_idx[label]] = 1.0
    return proba


def _uniform_proba(n: int) -> np.ndarray:
    """Return an (n, 3) array of uniform 1/3 probabilities."""
    return np.full((n, 3), 1.0 / 3.0)


def _make_matches_df(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Build a minimal matches DataFrame spanning 10 years."""
    rng = np.random.default_rng(seed)
    teams = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
    dates = pd.date_range("2010-01-01", periods=n, freq="30D")
    rows = []
    for dt in dates:
        ht = rng.choice(teams)
        at = rng.choice([t for t in teams if t != ht])
        hg = int(rng.integers(0, 4))
        ag = int(rng.integers(0, 4))
        rows.append(
            {
                "date": dt,
                "home_team": ht,
                "away_team": at,
                "home_goals": hg,
                "away_goals": ag,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# metrics.py — accuracy
# ---------------------------------------------------------------------------

class TestAccuracy:
    def test_accuracy_all_correct(self):
        labels = ["H", "D", "A", "H", "A"]
        y_true = pd.Series(labels)
        proba = _perfect_proba(labels)
        assert accuracy(y_true, proba) == pytest.approx(1.0)

    def test_accuracy_all_wrong(self):
        """When every prediction is the wrong class, accuracy is 0."""
        y_true = pd.Series(["H", "H", "H"])
        # Always predict A (index 0)
        proba = np.array([[1.0, 0.0, 0.0]] * 3)  # argmax = A, true = H
        assert accuracy(y_true, proba) == pytest.approx(0.0)

    def test_accuracy_partial(self):
        y_true = pd.Series(["H", "D", "A"])
        # First row correct (H=index 2), rest wrong
        proba = np.array([
            [0.0, 0.0, 1.0],  # predicts H → correct
            [1.0, 0.0, 0.0],  # predicts A → wrong (true D)
            [0.0, 1.0, 0.0],  # predicts D → wrong (true A)
        ])
        assert accuracy(y_true, proba) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# metrics.py — log_loss_score
# ---------------------------------------------------------------------------

class TestLogLoss:
    def test_log_loss_perfect_predictions(self):
        """Perfect predictions should give a very small log loss (not necessarily 0
        due to sklearn clipping, but < 0.01)."""
        labels = ["H", "D", "A", "H"]
        y_true = pd.Series(labels)
        proba = _perfect_proba(labels)
        # sklearn clips proba, so log_loss is tiny but nonzero
        assert log_loss_score(y_true, proba) < 0.01

    def test_log_loss_uniform_is_large(self):
        """Uniform probabilities should give log loss ≈ log(3) ≈ 1.099."""
        n = 100
        y_true = pd.Series(["H"] * n)
        proba = _uniform_proba(n)
        ll = log_loss_score(y_true, proba)
        assert abs(ll - np.log(3)) < 0.01


# ---------------------------------------------------------------------------
# metrics.py — brier_score
# ---------------------------------------------------------------------------

class TestBrierScore:
    def test_brier_score_perfect_predictions(self):
        """Perfect predictions → Brier score = 0."""
        labels = ["H", "D", "A"]
        y_true = pd.Series(labels)
        proba = _perfect_proba(labels)
        assert brier_score(y_true, proba) == pytest.approx(0.0, abs=1e-12)

    def test_brier_score_worst_case(self):
        """Probability = 1 on the WRONG class → max Brier per row = 2."""
        y_true = pd.Series(["H"])  # true = H (index 2)
        # Predict A with certainty (index 0)
        proba = np.array([[1.0, 0.0, 0.0]])
        # (1-0)^2 + (0-0)^2 + (0-1)^2 = 2
        assert brier_score(y_true, proba) == pytest.approx(2.0)

    def test_brier_score_uniform(self):
        """Uniform probabilities → known Brier score."""
        y_true = pd.Series(["H"])
        proba = np.array([[1 / 3, 1 / 3, 1 / 3]])
        # (1/3)^2 + (1/3 - 0)^2 + (1/3 - 1)^2 = 1/9 + 1/9 + 4/9 = 6/9 = 2/3
        assert brier_score(y_true, proba) == pytest.approx(2 / 3, abs=1e-10)


# ---------------------------------------------------------------------------
# metrics.py — ranked_probability_score
# ---------------------------------------------------------------------------

class TestRPS:
    def test_rps_perfect_predictions(self):
        """Perfect predictions → RPS = 0."""
        labels = ["H", "D", "A", "H"]
        y_true = pd.Series(labels)
        proba = _perfect_proba(labels)
        assert ranked_probability_score(y_true, proba) == pytest.approx(0.0, abs=1e-12)

    def test_rps_adjacent_outcome(self):
        """RPS penalises close misses less than distant ones."""
        # True = H (index 2, rightmost); predict D (index 1, adjacent) vs A (index 0)
        y_true_h = pd.Series(["H"])
        proba_near = np.array([[0.0, 1.0, 0.0]])  # predict D
        proba_far = np.array([[1.0, 0.0, 0.0]])   # predict A
        rps_near = ranked_probability_score(y_true_h, proba_near)
        rps_far = ranked_probability_score(y_true_h, proba_far)
        assert rps_near < rps_far

    def test_rps_known_value(self):
        """Manual calculation for a single match with known answer."""
        # True = H: CDF_true = [0, 0, 1]
        # Predict: A=0.5, D=0.3, H=0.2 → CDF_pred = [0.5, 0.8, 1.0]
        # RPS = (0.5-0)^2 + (0.8-0)^2 = 0.25 + 0.64 = 0.89
        y_true = pd.Series(["H"])
        proba = np.array([[0.5, 0.3, 0.2]])
        expected = (0.5 - 0.0) ** 2 + (0.8 - 0.0) ** 2
        assert ranked_probability_score(y_true, proba) == pytest.approx(expected, abs=1e-10)


# ---------------------------------------------------------------------------
# metrics.py — expected_calibration_error
# ---------------------------------------------------------------------------

class TestECE:
    def test_ece_perfect_predictions(self):
        """Perfect calibration (perfectly predicted outcomes) → ECE ≈ 0."""
        labels = ["H"] * 10 + ["D"] * 10 + ["A"] * 10
        y_true = pd.Series(labels)
        proba = _perfect_proba(labels)
        ece = expected_calibration_error(y_true, proba)
        assert ece == pytest.approx(0.0, abs=1e-10)

    def test_ece_returns_float(self):
        y_true = pd.Series(["H", "D", "A"])
        proba = _uniform_proba(3)
        ece = expected_calibration_error(y_true, proba)
        assert isinstance(ece, float)
        assert ece >= 0.0


# ---------------------------------------------------------------------------
# metrics.py — compute_all_metrics
# ---------------------------------------------------------------------------

class TestComputeAllMetrics:
    def test_compute_all_metrics_returns_all_keys(self):
        labels = ["H", "D", "A", "H", "D"]
        y_true = pd.Series(labels)
        proba = _perfect_proba(labels)
        result = compute_all_metrics(y_true, proba)
        assert set(result.keys()) == {"accuracy", "log_loss", "brier_score", "rps", "ece"}

    def test_compute_all_metrics_values_are_floats(self):
        y_true = pd.Series(["H", "D", "A"])
        proba = _uniform_proba(3)
        result = compute_all_metrics(y_true, proba)
        for key, val in result.items():
            assert isinstance(val, float), f"{key} is not a float"


# ---------------------------------------------------------------------------
# backtesting.py — BacktestResult dataclass
# ---------------------------------------------------------------------------

class TestBacktestResult:
    def test_backtest_result_is_dataclass(self):
        """BacktestResult must be a dataclass with the expected fields."""
        field_names = {f.name for f in fields(BacktestResult)}
        assert field_names == {"window_start", "window_end", "train_size", "test_size", "metrics"}

    def test_backtest_result_stores_values(self):
        r = BacktestResult(
            window_start="2015-01-01",
            window_end="2015-07-01",
            train_size=500,
            test_size=30,
            metrics={"accuracy": 0.5},
        )
        assert r.window_start == "2015-01-01"
        assert r.train_size == 500
        assert r.metrics["accuracy"] == 0.5


# ---------------------------------------------------------------------------
# backtesting.py — RollingBacktester
# ---------------------------------------------------------------------------

class TestRollingBacktester:
    """Use a mocked XGBoostMatchModel to avoid training cost."""

    def _make_backtester(self, matches_df: pd.DataFrame) -> RollingBacktester:
        elo_model = MagicMock()
        elo_model.config = MagicMock()
        poisson_model = MagicMock()
        poisson_model.config = MagicMock()
        return RollingBacktester(
            matches_df=matches_df,
            elo_model=elo_model,
            poisson_model=poisson_model,
            min_train_years=5,
            test_window_months=6,
            step_months=6,
        )

    def test_rolling_backtester_no_shuffle(self):
        """The backtester must not shuffle data — training always precedes testing."""
        matches_df = _make_matches_df(n=120)
        backtester = self._make_backtester(matches_df)
        # Dates must remain in ascending order; we verify the internal _matches
        dates = backtester._matches["date"].values
        assert all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1))

    def test_rolling_backtester_windows_are_chronological(self):
        """window_start of each result must be strictly increasing."""
        matches_df = _make_matches_df(n=200, seed=7)

        # Patch XGBoostMatchModel to avoid real training
        import src.evaluation.backtesting as bt_module
        from unittest.mock import patch, MagicMock
        import numpy as np

        fake_xgb = MagicMock()
        fake_xgb.predict_proba.return_value = np.array([[0.5, 0.25, 0.25]])
        fake_xgb.classes_ = ["A", "D", "H"]

        elo = MagicMock()
        elo.config = MagicMock()
        elo.ratings = {}
        poisson = MagicMock()
        poisson.config = MagicMock()

        backtester = RollingBacktester(
            matches_df=matches_df,
            elo_model=elo,
            poisson_model=poisson,
            min_train_years=3,
            test_window_months=6,
            step_months=6,
        )

        with patch("src.evaluation.backtesting.EloModel") as MockElo, \
             patch("src.evaluation.backtesting.PoissonModel") as MockPoisson, \
             patch("src.evaluation.backtesting.FeatureBuilder") as MockFB, \
             patch("src.evaluation.backtesting.XGBoostMatchModel") as MockXGB:

            # Configure mock EloModel
            mock_elo_inst = MagicMock()
            MockElo.return_value = mock_elo_inst

            # Configure mock PoissonModel
            mock_poi_inst = MagicMock()
            MockPoisson.return_value = mock_poi_inst

            # Configure mock FeatureBuilder
            fake_feat = {col: 0.0 for col in FEATURE_COLUMNS}
            mock_fb_inst = MagicMock()
            mock_fb_inst.build_training_matrix.return_value = pd.DataFrame(
                {col: [0.0] * 30 for col in FEATURE_COLUMNS}
                | {"label": ["H"] * 10 + ["D"] * 10 + ["A"] * 10}
            )
            mock_fb_inst.build_features_for_match.return_value = fake_feat
            MockFB.return_value = mock_fb_inst

            # Configure mock XGBoostMatchModel
            mock_xgb_inst = MagicMock()
            mock_xgb_inst.predict_proba.return_value = np.tile(
                [0.5, 0.25, 0.25], (50, 1)
            )
            MockXGB.return_value = mock_xgb_inst

            results = backtester.run()

        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert results[i].window_start < results[i + 1].window_start

    def test_rolling_backtester_to_dataframe(self):
        """to_dataframe should return a DataFrame with expected columns."""
        results = [
            BacktestResult(
                window_start="2015-01-01",
                window_end="2015-07-01",
                train_size=100,
                test_size=20,
                metrics={"accuracy": 0.5, "log_loss": 1.0, "brier_score": 0.4, "rps": 0.3, "ece": 0.1},
            )
        ]
        backtester = self._make_backtester(_make_matches_df())
        df = backtester.to_dataframe(results)
        assert "window_start" in df.columns
        assert "accuracy" in df.columns
        assert len(df) == 1


# ---------------------------------------------------------------------------
# benchmark.py — ModelBenchmarkResult dataclass
# ---------------------------------------------------------------------------

class TestModelBenchmarkResult:
    def test_model_benchmark_result_is_dataclass(self):
        field_names = {f.name for f in fields(ModelBenchmarkResult)}
        assert field_names == {"model_name", "metrics"}

    def test_model_benchmark_result_stores_values(self):
        r = ModelBenchmarkResult(
            model_name="Elo",
            metrics={"accuracy": 0.45, "log_loss": 1.05},
        )
        assert r.model_name == "Elo"
        assert r.metrics["log_loss"] == 1.05


# ---------------------------------------------------------------------------
# benchmark.py — ModelBenchmarker
# ---------------------------------------------------------------------------

def _make_benchmarker(matches_df: pd.DataFrame | None = None) -> ModelBenchmarker:
    """Build a ModelBenchmarker with fully mocked component models."""
    if matches_df is None:
        matches_df = _make_matches_df(n=80, seed=1)

    elo = MagicMock()
    elo.win_draw_loss_probabilities.return_value = (0.5, 0.25, 0.25)

    poisson = MagicMock()
    poisson.win_draw_loss_from_poisson.return_value = (0.5, 0.25, 0.25)

    xgb = MagicMock()
    # Return shape (n, 3) matching the number of rows passed in
    xgb.predict_proba.side_effect = lambda X: np.tile([0.4, 0.3, 0.3], (len(X), 1))

    feature_builder = MagicMock()
    feat = {col: 0.0 for col in FEATURE_COLUMNS}
    feature_builder.build_features_for_match.return_value = feat

    return ModelBenchmarker(
        matches_df=matches_df,
        elo_model=elo,
        poisson_model=poisson,
        xgb_model=xgb,
        feature_builder=feature_builder,
        test_split_date="2015-01-01",
    )


class TestModelBenchmarker:
    def test_model_benchmarker_returns_all_four_models(self):
        """run() must return exactly 4 results covering all model names."""
        benchmarker = _make_benchmarker()
        results = benchmarker.run()
        model_names = {r.model_name for r in results}
        assert "Elo" in model_names
        assert "Poisson" in model_names
        assert "XGBoost" in model_names
        # Ensemble name includes weight annotation
        ensemble_names = [n for n in model_names if "Ensemble" in n]
        assert len(ensemble_names) == 1

    def test_benchmarker_best_model_returns_lowest_log_loss(self):
        """best_model() must return the name of the model with the lowest log_loss."""
        results = [
            ModelBenchmarkResult("Elo", {"log_loss": 1.2, "accuracy": 0.4}),
            ModelBenchmarkResult("Poisson", {"log_loss": 1.0, "accuracy": 0.45}),
            ModelBenchmarkResult("XGBoost", {"log_loss": 0.95, "accuracy": 0.5}),
            ModelBenchmarkResult("Ensemble (0.3/0.3/0.4)", {"log_loss": 0.93, "accuracy": 0.51}),
        ]
        benchmarker = _make_benchmarker()
        best = benchmarker.best_model(results)
        assert best == "Ensemble (0.3/0.3/0.4)"

    def test_benchmarker_to_dataframe_shape(self):
        """to_dataframe must return a DataFrame with 4 rows and expected metric columns."""
        benchmarker = _make_benchmarker()
        results = benchmarker.run()
        df = benchmarker.to_dataframe(results)
        assert len(df) == 4
        assert "accuracy" in df.columns
        assert "log_loss" in df.columns

    def test_benchmarker_save_creates_csv_and_md(self):
        """save() must create benchmark_results.csv and benchmark_summary.md."""
        benchmarker = _make_benchmarker()
        results = benchmarker.run()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path, md_path = benchmarker.save(results, tmp_path)

            assert csv_path.exists()
            assert md_path.exists()
            assert csv_path.suffix == ".csv"
            assert md_path.suffix == ".md"

            # Verify CSV is non-empty and has a header
            content = csv_path.read_text()
            assert "accuracy" in content

            # Verify markdown contains model names
            md_content = md_path.read_text()
            assert "Elo" in md_content or "Model" in md_content

    def test_benchmarker_best_model_raises_on_empty(self):
        """best_model() must raise ValueError when given an empty list."""
        benchmarker = _make_benchmarker()
        with pytest.raises(ValueError, match="empty"):
            benchmarker.best_model([])

    def test_benchmarker_all_metrics_keys_present(self):
        """Each result's metrics dict must contain all 5 standard metric keys."""
        benchmarker = _make_benchmarker()
        results = benchmarker.run()
        expected_keys = {"accuracy", "log_loss", "brier_score", "rps", "ece"}
        for r in results:
            assert set(r.metrics.keys()) == expected_keys
