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
