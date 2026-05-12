"""Tests for src/scenarios/scenario_runner.py, upset_analysis.py, and stress_tests.py."""
from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.scenarios.scenario_runner import ScenarioRunner, Perturbation


def _make_elo(rating: float = 1600.0) -> MagicMock:
    elo = MagicMock()
    elo.ratings = {"Brazil": rating, "France": 1550.0}
    elo.win_draw_loss_probabilities.return_value = (0.50, 0.25, 0.25)
    return elo


def _make_poisson() -> MagicMock:
    poisson = MagicMock()
    poisson.attack_strength = {"Brazil": 1.2, "France": 1.1}
    poisson.defense_strength = {"Brazil": 0.9, "France": 0.95}
    poisson.win_draw_loss_from_poisson.return_value = (0.50, 0.25, 0.25)
    return poisson


def _make_xgb() -> MagicMock:
    xgb = MagicMock()
    xgb.predict_proba_dict.return_value = [
        {"home_win": 0.50, "draw": 0.25, "away_win": 0.25}
    ]
    return xgb


def _make_feature_builder() -> MagicMock:
    from src.features.feature_builder import FEATURE_COLUMNS
    fb = MagicMock()
    fb.build_features_for_match.return_value = {col: 0.5 for col in FEATURE_COLUMNS}
    return fb


def _make_runner() -> ScenarioRunner:
    return ScenarioRunner(
        elo_model=_make_elo(),
        poisson_model=_make_poisson(),
        xgb_model=_make_xgb(),
        feature_builder=_make_feature_builder(),
        weights=(0.3, 0.3, 0.4),
    )


class TestPerturbation:
    def test_valid_perturbation(self):
        p = Perturbation(param="attack_strength", team="France", delta=-0.3)
        assert p.param == "attack_strength"
        assert p.team == "France"
        assert p.delta == -0.3

    def test_invalid_param_raises(self):
        with pytest.raises(ValueError, match="Unknown param"):
            Perturbation(param="invalid_stat", team="Brazil", delta=0.1).validate()


class TestScenarioRunner:
    def test_run_returns_required_keys(self):
        runner = _make_runner()
        p = Perturbation(param="attack_strength", team="France", delta=-0.3)
        result = runner.run("Brazil", "France", [p])
        assert "home_team" in result
        assert "away_team" in result
        assert "baseline" in result
        assert "perturbed" in result
        assert "delta" in result
        assert "perturbations" in result

    def test_delta_structure(self):
        runner = _make_runner()
        p = Perturbation(param="attack_strength", team="France", delta=-0.3)
        result = runner.run("Brazil", "France", [p])
        for key in ("home_win", "draw", "away_win"):
            assert key in result["delta"]

    def test_baseline_probs_sum_to_one(self):
        runner = _make_runner()
        result = runner.run("Brazil", "France", [])
        total = sum(result["baseline"].values())
        assert abs(total - 1.0) < 0.01

    def test_perturbed_probs_sum_to_one(self):
        runner = _make_runner()
        p = Perturbation(param="attack_strength", team="France", delta=-0.3)
        result = runner.run("Brazil", "France", [p])
        total = sum(result["perturbed"].values())
        assert abs(total - 1.0) < 0.01

    def test_no_perturbations_gives_zero_delta(self):
        runner = _make_runner()
        result = runner.run("Brazil", "France", [])
        for v in result["delta"].values():
            assert abs(v) < 1e-6

    def test_to_markdown_returns_string(self):
        runner = _make_runner()
        p = Perturbation(param="attack_strength", team="France", delta=-0.3)
        result = runner.run("Brazil", "France", [p])
        md = runner.to_markdown(result)
        assert isinstance(md, str)
        assert "Brazil" in md
        assert "France" in md

    def test_state_restored_on_blend_failure(self):
        """Elo ratings must be restored even if _blend raises."""
        runner = _make_runner()
        original_rating = runner._elo.ratings.get("France", 1550.0)
        # Make the second blend call fail
        call_count = [0]
        original_blend = runner._blend
        def failing_blend(home, away, date, form_deltas=None):
            call_count[0] += 1
            if call_count[0] == 2:  # second call (perturbed) fails
                raise RuntimeError("simulated blend failure")
            return original_blend(home, away, date, form_deltas=form_deltas)
        runner._blend = failing_blend

        with pytest.raises(RuntimeError, match="simulated blend failure"):
            runner.run("Brazil", "France", [Perturbation(param="elo_rating", team="France", delta=100.0)])

        assert runner._elo.ratings.get("France") == original_rating


from src.scenarios.upset_analysis import UpsetAnalyzer, UpsetFlag


class TestUpsetAnalyzer:
    def test_clear_favourite_not_flagged(self):
        analyzer = UpsetAnalyzer()
        flag = analyzer.check(
            home_team="Brazil", away_team="Andorra",
            home_win_prob=0.85, away_win_prob=0.05,
            confidence_band="High", stability_band="Stable",
        )
        assert not flag.is_upset

    def test_underdog_above_threshold_is_flagged(self):
        analyzer = UpsetAnalyzer(underdog_threshold=0.35)
        flag = analyzer.check(
            home_team="Brazil", away_team="France",
            home_win_prob=0.40, away_win_prob=0.38,
            confidence_band="High", stability_band="Stable",
        )
        assert flag.is_upset

    def test_low_confidence_triggers_flag(self):
        analyzer = UpsetAnalyzer()
        flag = analyzer.check(
            home_team="Brazil", away_team="France",
            home_win_prob=0.60, away_win_prob=0.20,
            confidence_band="Low", stability_band="Stable",
        )
        assert flag.is_upset

    def test_unstable_prediction_triggers_flag(self):
        analyzer = UpsetAnalyzer()
        flag = analyzer.check(
            home_team="Brazil", away_team="France",
            home_win_prob=0.60, away_win_prob=0.20,
            confidence_band="High", stability_band="Unstable",
        )
        assert flag.is_upset

    def test_flag_has_reason(self):
        analyzer = UpsetAnalyzer(underdog_threshold=0.35)
        flag = analyzer.check(
            home_team="Brazil", away_team="France",
            home_win_prob=0.40, away_win_prob=0.38,
            confidence_band="High", stability_band="Stable",
        )
        assert flag.is_upset
        assert len(flag.reasons) > 0


from src.scenarios.stress_tests import RobustnessTester, StressTestResult
from src.features.feature_builder import FEATURE_COLUMNS


class TestRobustnessTester:
    def _make_tester(self) -> RobustnessTester:
        xgb = MagicMock()
        xgb.predict_proba_dict.return_value = [
            {"home_win": 0.50, "draw": 0.25, "away_win": 0.25}
        ]
        xgb.feature_names = FEATURE_COLUMNS
        return RobustnessTester(xgb_model=xgb, random_seed=42)

    def _base_features(self) -> dict:
        return {col: 0.5 for col in FEATURE_COLUMNS}

    def test_run_all_returns_four_results(self):
        tester = self._make_tester()
        results = tester.run_all(self._base_features())
        assert len(results) == 4

    def test_all_results_are_stress_test_result(self):
        tester = self._make_tester()
        results = tester.run_all(self._base_features())
        assert all(isinstance(r, StressTestResult) for r in results)

    def test_all_tests_pass_with_valid_model(self):
        tester = self._make_tester()
        results = tester.run_all(self._base_features())
        failed = [r for r in results if not r.passed]
        assert not failed, f"Stress tests failed: {[r.error for r in failed]}"

    def test_to_markdown_contains_pass_or_fail(self):
        tester = self._make_tester()
        results = tester.run_all(self._base_features())
        md = tester.to_markdown(results)
        assert isinstance(md, str)
        assert "PASS" in md or "FAIL" in md

    def test_result_test_names_are_descriptive(self):
        tester = self._make_tester()
        results = tester.run_all(self._base_features())
        for r in results:
            assert len(r.test_name) > 0
