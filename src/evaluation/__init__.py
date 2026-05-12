"""Evaluation package — metrics, rolling backtest, and model benchmarking."""

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

__all__ = [
    "CLASSES",
    "accuracy",
    "brier_score",
    "compute_all_metrics",
    "expected_calibration_error",
    "log_loss_score",
    "ranked_probability_score",
    "BacktestResult",
    "RollingBacktester",
    "ModelBenchmarkResult",
    "ModelBenchmarker",
]
