"""Tests for src/explainability/shap_engine.py.

shap.TreeExplainer is always mocked — tests pass whether or not shap is installed,
and explicitly test graceful degradation.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.features.feature_builder import FEATURE_COLUMNS


def _make_xgb_model() -> MagicMock:
    model = MagicMock()
    model.feature_names = FEATURE_COLUMNS
    return model


def _shap_values_fixture(n: int = 5) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, len(FEATURE_COLUMNS), 3))


class TestSHAPUnavailable:
    def test_local_shap_raises_import_error_when_unavailable(self):
        with patch("src.explainability.shap_engine._SHAP_AVAILABLE", False):
            from src.explainability.shap_engine import SHAPEngine
            engine = SHAPEngine.__new__(SHAPEngine)
            engine._model = _make_xgb_model()
            engine._explainer = None
            with pytest.raises(ImportError, match="shap"):
                engine.local_shap(pd.DataFrame([{col: 0.5 for col in FEATURE_COLUMNS}]))

    def test_global_shap_raises_import_error_when_unavailable(self):
        with patch("src.explainability.shap_engine._SHAP_AVAILABLE", False):
            from src.explainability.shap_engine import SHAPEngine
            engine = SHAPEngine.__new__(SHAPEngine)
            engine._model = _make_xgb_model()
            engine._explainer = None
            with pytest.raises(ImportError, match="shap"):
                engine.global_shap(pd.DataFrame([{col: 0.5 for col in FEATURE_COLUMNS}]))


class TestSHAPEngineWithMockedSHAP:
    def _make_engine(self, n: int = 5):
        from src.explainability.shap_engine import SHAPEngine
        xgb = _make_xgb_model()
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = _shap_values_fixture(n=n)
        engine = SHAPEngine.__new__(SHAPEngine)
        engine._model = xgb
        engine._explainer = mock_explainer
        return engine

    def test_global_shap_returns_dict_with_feature_names(self):
        with patch("src.explainability.shap_engine._SHAP_AVAILABLE", True):
            engine = self._make_engine()
            X = pd.DataFrame([{col: 0.5 for col in FEATURE_COLUMNS}] * 5)
            result = engine.global_shap(X)
            assert isinstance(result, dict)
            assert all(col in result for col in FEATURE_COLUMNS)

    def test_global_shap_values_non_negative(self):
        with patch("src.explainability.shap_engine._SHAP_AVAILABLE", True):
            engine = self._make_engine()
            X = pd.DataFrame([{col: 0.5 for col in FEATURE_COLUMNS}] * 5)
            result = engine.global_shap(X)
            assert all(v >= 0 for v in result.values())

    def test_local_shap_returns_list(self):
        with patch("src.explainability.shap_engine._SHAP_AVAILABLE", True):
            engine = self._make_engine(n=1)
            X = pd.DataFrame([{col: 0.5 for col in FEATURE_COLUMNS}])
            result = engine.local_shap(X)
            assert isinstance(result, list)
            assert len(result) > 0

    def test_local_shap_entries_have_feature_and_value(self):
        with patch("src.explainability.shap_engine._SHAP_AVAILABLE", True):
            engine = self._make_engine(n=1)
            X = pd.DataFrame([{col: 0.5 for col in FEATURE_COLUMNS}])
            result = engine.local_shap(X)
            for entry in result:
                assert "feature" in entry
                assert "shap_value" in entry


class TestFeatureImpactRanker:
    def _make_engine_with_global(self, values: dict):
        from src.explainability.shap_engine import SHAPEngine
        engine = SHAPEngine.__new__(SHAPEngine)
        engine._model = _make_xgb_model()
        mock_explainer = MagicMock()
        arr = np.zeros((3, len(FEATURE_COLUMNS), 3))
        for i, col in enumerate(FEATURE_COLUMNS):
            arr[:, i, :] = values.get(col, 0.0)
        mock_explainer.shap_values.return_value = arr
        engine._explainer = mock_explainer
        return engine

    def test_rank_returns_list_sorted_by_importance(self):
        from src.explainability.feature_impact import FeatureImpactRanker
        with patch("src.explainability.shap_engine._SHAP_AVAILABLE", True):
            vals = {col: float(i) for i, col in enumerate(FEATURE_COLUMNS)}
            engine = self._make_engine_with_global(vals)
            ranker = FeatureImpactRanker(engine)
            X = pd.DataFrame([{col: 1.0 for col in FEATURE_COLUMNS}] * 3)
            ranked = ranker.rank(X, top_n=5)
            assert len(ranked) == 5
            assert ranked[0]["rank"] == 1
            importances = [r["importance"] for r in ranked]
            assert importances == sorted(importances, reverse=True)

    def test_to_markdown_contains_feature_names(self):
        from src.explainability.feature_impact import FeatureImpactRanker
        ranker = FeatureImpactRanker(MagicMock())
        ranked = [{"rank": 1, "feature": "elo_diff", "importance": 0.18}]
        md = ranker.to_markdown(ranked)
        assert "elo_diff" in md
        assert "0.180000" in md


class TestPredictionExplainer:
    def _make_engine_with_local(self):
        from src.explainability.shap_engine import SHAPEngine
        engine = SHAPEngine.__new__(SHAPEngine)
        engine._model = _make_xgb_model()
        arr = np.zeros((1, len(FEATURE_COLUMNS), 3))
        arr[0, 0, 2] = 0.5  # first feature has some SHAP for class H (index 2)
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = arr
        engine._explainer = mock_explainer
        return engine

    def test_explain_returns_top_drivers(self):
        from src.explainability.prediction_explainer import PredictionExplainer
        with patch("src.explainability.shap_engine._SHAP_AVAILABLE", True):
            engine = self._make_engine_with_local()
            explainer = PredictionExplainer(engine)
            X = pd.DataFrame([{col: 1.0 for col in FEATURE_COLUMNS}])
            result = explainer.explain("Brazil", "France", X, outcome="H", top_n=3)
            assert result["home_team"] == "Brazil"
            assert result["away_team"] == "France"
            assert result["outcome_explained"] == "H"
            assert len(result["top_drivers"]) == 3
            assert result["top_drivers"][0]["rank"] == 1

    def test_to_markdown_contains_teams(self):
        from src.explainability.prediction_explainer import PredictionExplainer
        explainer = PredictionExplainer(MagicMock())
        explanation = {
            "home_team": "Brazil",
            "away_team": "France",
            "outcome_explained": "H",
            "top_drivers": [{"rank": 1, "feature": "elo_diff", "shap_value": 0.15, "raw_value": 50.0}],
        }
        md = explainer.to_markdown(explanation)
        assert "Brazil" in md
        assert "France" in md
        assert "elo_diff" in md

    def test_save_creates_md_and_json(self, tmp_path):
        from src.explainability.prediction_explainer import PredictionExplainer
        explainer = PredictionExplainer(MagicMock())
        explanation = {
            "home_team": "Brazil",
            "away_team": "France",
            "outcome_explained": "H",
            "top_drivers": [{"rank": 1, "feature": "elo_diff", "shap_value": 0.15, "raw_value": 50.0}],
        }
        explainer.save(explanation, output_dir=tmp_path, match_date="2026-06-01")
        assert (tmp_path / "Brazil_vs_France_2026-06-01.md").exists()
        assert (tmp_path / "Brazil_vs_France_2026-06-01.json").exists()
