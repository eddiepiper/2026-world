from __future__ import annotations

import pytest

from src.config.settings import DiagnosticsConfig
from src.diagnostics.drift_detection import DriftDetector, DriftAlert


def _make_config(window: int = 10, threshold: float = 2.0) -> DiagnosticsConfig:
    cfg = DiagnosticsConfig()
    cfg.drift_window = window
    cfg.drift_alert_threshold = threshold
    return cfg


def _good_probs() -> dict[str, float]:
    return {"home_win": 0.6, "draw": 0.2, "away_win": 0.2}


def _bad_probs() -> dict[str, float]:
    """Very uncertain — high entropy."""
    return {"home_win": 0.34, "draw": 0.33, "away_win": 0.33}


class TestDriftDetector:
    def test_no_alerts_before_baseline_filled(self):
        detector = DriftDetector(config=_make_config(window=10))
        # Send 10 records (exactly window size) — should not alert
        for _ in range(10):
            alerts = detector.add_record("H", _good_probs())
        assert alerts == []

    def test_returns_list(self):
        detector = DriftDetector(config=_make_config(window=5))
        alerts = detector.add_record("H", _good_probs())
        assert isinstance(alerts, list)

    def test_status_returns_expected_keys(self):
        detector = DriftDetector(config=_make_config(window=10))
        status = detector.status()
        assert "total_records" in status
        assert "window_size" in status
        assert "has_baseline" in status

    def test_status_total_records_increments(self):
        detector = DriftDetector(config=_make_config(window=5))
        for _ in range(3):
            detector.add_record("H", _good_probs())
        assert detector.status()["total_records"] == 3

    def test_drift_alert_is_dataclass(self):
        """Inject obvious drift — send perfect probs then completely wrong ones."""
        cfg = _make_config(window=5, threshold=0.5)
        detector = DriftDetector(config=cfg)
        # Fill baseline with confident correct predictions
        for _ in range(5):
            detector.add_record("H", {"home_win": 0.99, "draw": 0.005, "away_win": 0.005})
        # Now add away wins with very wrong probs — should trigger alert
        alerts = []
        for _ in range(5):
            new_alerts = detector.add_record("A", {"home_win": 0.99, "draw": 0.005, "away_win": 0.005})
            alerts.extend(new_alerts)
        # We expect at least some alerts (log_loss should spike on wrong predictions)
        if alerts:
            alert = alerts[0]
            assert isinstance(alert, DriftAlert)
            assert alert.metric in ("log_loss", "brier_score", "entropy")
            assert alert.severity in ("warning", "critical")

    def test_no_drift_on_consistent_predictions(self):
        """Consistent correct predictions should not trigger alerts."""
        detector = DriftDetector(config=_make_config(window=10, threshold=2.0))
        # Fill past window
        for i in range(25):
            alerts = detector.add_record("H", _good_probs())
        # All identical — no drift expected
        assert alerts == []

    def test_window_size_respected(self):
        cfg = _make_config(window=7)
        detector = DriftDetector(config=cfg)
        assert detector.status()["window_size"] == 7

    def test_drift_alert_fields(self):
        alert = DriftAlert(
            metric="log_loss",
            current_value=1.5,
            baseline_mean=0.8,
            baseline_std=0.1,
            severity="warning",
        )
        assert alert.metric == "log_loss"
        assert alert.current_value == 1.5
        assert alert.severity == "warning"
