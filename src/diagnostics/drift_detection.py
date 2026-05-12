from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np
from loguru import logger

from src.config.settings import DiagnosticsConfig, settings


@dataclass
class DriftAlert:
    metric: str          # "log_loss" | "brier_score" | "entropy"
    current_value: float
    baseline_mean: float
    baseline_std: float
    severity: Literal["warning", "critical"]


class DriftDetector:
    """Monitors rolling prediction quality and alerts on statistical drift.

    Usage:
        detector = DriftDetector()
        alerts = detector.add_record(y_true="H", probs={"home_win": 0.6, "draw": 0.2, "away_win": 0.2})
    """

    def __init__(self, config: DiagnosticsConfig | None = None) -> None:
        cfg = config or settings.diagnostics
        self._window = cfg.drift_window
        self._threshold = cfg.drift_alert_threshold
        # Store (y_true, probs) tuples — rolling window
        self._records: deque[tuple[str, dict[str, float]]] = deque(maxlen=cfg.drift_window)
        self._all_records: list[tuple[str, dict[str, float]]] = []

    def add_record(
        self, y_true: str, probs: dict[str, float]
    ) -> list[DriftAlert]:
        """Add a (prediction, outcome) pair and return any drift alerts.

        Returns an empty list if fewer than window + 1 records have been added
        (need baseline window + at least one more record to detect drift).
        """
        self._all_records.append((y_true, probs))
        self._records.append((y_true, probs))

        # Need at least window + 1 records total before we can compare
        if len(self._all_records) < self._window + 1:
            return []

        return self._check_drift()

    def status(self) -> dict:
        """Return current drift status summary."""
        n = len(self._all_records)
        return {
            "total_records": n,
            "window_size": self._window,
            "has_baseline": n >= self._window,
            "current_window_size": min(n, self._window),
        }

    def _check_drift(self) -> list[DriftAlert]:
        """Compare most recent window against baseline window."""
        baseline = self._all_records[:self._window]
        current = list(self._records)[-self._window:]

        current_metrics = self._compute_window_metrics(current)

        # Compute per-record metrics for the baseline to get mean and std.
        baseline_per_record = self._per_record_metrics(baseline)

        alerts = []
        for metric_name, current_val in current_metrics.items():
            baseline_vals = baseline_per_record[metric_name]
            baseline_mean = float(np.mean(baseline_vals))
            baseline_std = float(np.std(baseline_vals)) if len(baseline_vals) > 1 else 0.0

            if baseline_std < 1e-8:
                continue  # no variance to compare against

            z_score = (current_val - baseline_mean) / baseline_std
            if z_score > self._threshold:
                severity: Literal["warning", "critical"] = (
                    "critical" if z_score > self._threshold * 1.5 else "warning"
                )
                alert = DriftAlert(
                    metric=metric_name,
                    current_value=round(current_val, 6),
                    baseline_mean=round(baseline_mean, 6),
                    baseline_std=round(baseline_std, 6),
                    severity=severity,
                )
                alerts.append(alert)
                logger.warning(
                    f"Drift detected [{severity}] — {metric_name}: "
                    f"current={current_val:.4f}, baseline_mean={baseline_mean:.4f}, "
                    f"z={z_score:.2f}"
                )

        return alerts

    def _compute_window_metrics(
        self, records: list[tuple[str, dict[str, float]]]
    ) -> dict[str, float]:
        """Aggregate metrics over a window of records."""
        if not records:
            return {}
        per = self._per_record_metrics(records)
        return {k: float(np.mean(v)) for k, v in per.items()}

    def _per_record_metrics(
        self, records: list[tuple[str, dict[str, float]]]
    ) -> dict[str, list[float]]:
        """Compute per-record log_loss, brier_score, and entropy."""
        ll_list: list[float] = []
        brier_list: list[float] = []
        entropy_list: list[float] = []

        for y_true, probs in records:
            hw = probs.get("home_win", 1 / 3)
            d = probs.get("draw", 1 / 3)
            aw = probs.get("away_win", 1 / 3)
            p_vec = [hw, d, aw]
            total = sum(p_vec)
            if total > 0:
                p_vec = [p / total for p in p_vec]

            # Map outcome to index: H=0, D=1, A=2
            true_idx = {"H": 0, "D": 1, "A": 2}.get(y_true, 0)
            p_true = max(p_vec[true_idx], 1e-12)
            ll_list.append(-math.log(p_true))

            # Brier score (multiclass): sum of squared differences
            true_one_hot = [0.0, 0.0, 0.0]
            true_one_hot[true_idx] = 1.0
            brier = sum((p - o) ** 2 for p, o in zip(p_vec, true_one_hot))
            brier_list.append(brier)

            # Prediction entropy
            entropy = -sum(p * math.log(max(p, 1e-12)) for p in p_vec if p > 0)
            entropy_list.append(entropy)

        return {"log_loss": ll_list, "brier_score": brier_list, "entropy": entropy_list}
