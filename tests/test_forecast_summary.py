"""Tests for src/reports/forecast_summary.py and markdown_reporter.py."""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.reports.markdown_reporter import MarkdownReporter
from src.reports.forecast_summary import ReportSection, generate_report


class TestMarkdownReporter:
    def test_confidence_section_contains_score_and_band(self):
        reporter = MarkdownReporter()
        confidence = {
            "confidence_score": 0.75,
            "confidence_band": "High",
            "factor_breakdown": {
                "model_agreement": 0.88,
                "calibration_quality": 0.80,
                "feature_completeness": 1.00,
                "historical_reliability": 0.65,
                "prediction_volatility": 0.70,
            },
        }
        md = reporter.confidence_section(confidence)
        assert "confidence" in md.lower()
        assert "High" in md
        assert "0.75" in md

    def test_drift_section_no_alerts(self):
        reporter = MarkdownReporter()
        drift = {"status": "ok", "total_records": 100, "alerts": []}
        md = reporter.drift_section(drift)
        assert "drift" in md.lower() or "Drift" in md
        assert "100" in md

    def test_drift_section_with_alert(self):
        reporter = MarkdownReporter()
        drift = {
            "status": "alert",
            "total_records": 100,
            "alerts": [
                {"metric": "log_loss", "current_value": 1.2,
                 "baseline_mean": 0.9, "severity": "warning"}
            ],
        }
        md = reporter.drift_section(drift)
        assert "log_loss" in md

    def test_confidence_section_returns_string(self):
        reporter = MarkdownReporter()
        md = reporter.confidence_section(
            {"confidence_score": 0.5, "confidence_band": "Medium", "factor_breakdown": {}}
        )
        assert isinstance(md, str)
        assert len(md) > 0

    def test_shap_section_no_shap(self):
        reporter = MarkdownReporter()
        md = reporter.shap_section(None)
        assert "SHAP" in md or "shap" in md.lower()

    def test_shap_section_with_data(self):
        reporter = MarkdownReporter()
        global_shap = {"elo_diff": 0.18, "home_form": 0.11, "away_form": 0.06}
        md = reporter.shap_section(global_shap, top_n=2)
        assert "elo_diff" in md
        assert isinstance(md, str)


class TestForecastSummary:
    def test_generate_report_creates_file(self):
        sections = [
            ReportSection(title="Confidence", content="Score: 0.75"),
            ReportSection(title="Drift", content="Status: OK"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out_path = generate_report(sections, output_dir=Path(tmp))
            assert out_path.exists()
            text = out_path.read_text()
            assert "Confidence" in text
            assert "Drift" in text

    def test_generate_report_filename_contains_forecast_report(self):
        sections = [ReportSection(title="Test", content="content")]
        with tempfile.TemporaryDirectory() as tmp:
            out_path = generate_report(sections, output_dir=Path(tmp))
            assert "forecast_report" in out_path.name

    def test_report_section_dataclass(self):
        section = ReportSection(title="My Section", content="body text")
        assert section.title == "My Section"
        assert section.content == "body text"


class TestMarkdownReporterExtended:
    def test_stability_section_contains_band(self):
        reporter = MarkdownReporter()
        stability = {
            "stability_band": "Stable",
            "home_team": "Brazil",
            "away_team": "France",
            "std_probabilities": {"home_win": 0.01, "draw": 0.005, "away_win": 0.008},
        }
        md = reporter.stability_section(stability)
        assert "Stable" in md
        assert "Brazil" in md

    def test_stability_section_empty_stds(self):
        reporter = MarkdownReporter()
        md = reporter.stability_section({"stability_band": "Moderate", "home_team": "A", "away_team": "B", "std_probabilities": {}})
        assert "Moderate" in md

    def test_benchmark_section_missing_file(self):
        reporter = MarkdownReporter()
        md = reporter.benchmark_section(Path("/nonexistent/path/benchmark.csv"))
        assert "not available" in md.lower() or "unavailable" in md.lower()

    def test_benchmark_section_with_csv(self, tmp_path):
        reporter = MarkdownReporter()
        csv_path = tmp_path / "benchmark_results.csv"
        csv_path.write_text(
            "model,accuracy,log_loss,brier_score\n"
            "Elo,0.61,0.88,0.52\n"
            "XGBoost,0.63,0.85,0.50\n"
        )
        md = reporter.benchmark_section(csv_path)
        assert "XGBoost" in md
        assert "0.8500" in md


class TestReliabilityMonitor:
    def _make_monitor(self):
        from src.diagnostics.reliability_monitor import ReliabilityMonitor
        confidence_scorer = MagicMock()
        confidence_scorer.score.return_value = {
            "confidence_score": 0.75,
            "confidence_band": "High",
            "factor_breakdown": {},
        }
        drift_detector = MagicMock()
        drift_detector.status.return_value = {"total_records": 10, "alerts": []}
        stability_analyzer = MagicMock()
        stability_analyzer.analyze.return_value = {
            "stability_band": "Stable",
            "home_team": "Brazil",
            "away_team": "France",
            "std_probabilities": {},
        }
        return ReliabilityMonitor(confidence_scorer, drift_detector, stability_analyzer)

    def test_report_returns_all_keys(self):
        monitor = self._make_monitor()
        result = monitor.report(
            home_team="Brazil",
            away_team="France",
            match_date=pd.Timestamp("2026-06-01"),
            component_models={"elo": {"home_win": 0.5, "draw": 0.25, "away_win": 0.25}},
            ensemble_probs={"home_win": 0.5, "draw": 0.25, "away_win": 0.25},
            feature_vector={"elo_diff": 100.0},
        )
        assert "confidence" in result
        assert "drift" in result
        assert "stability" in result

    def test_report_confidence_band(self):
        monitor = self._make_monitor()
        result = monitor.report(
            home_team="Brazil",
            away_team="France",
            match_date=None,
            component_models={},
            ensemble_probs={"home_win": 0.5, "draw": 0.25, "away_win": 0.25},
            feature_vector={},
        )
        assert result["confidence"]["confidence_band"] == "High"
