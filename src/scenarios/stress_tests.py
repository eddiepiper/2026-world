from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

from src.features.feature_builder import FEATURE_COLUMNS
from src.ml.xgboost_model import XGBoostMatchModel


@dataclass
class StressTestResult:
    test_name: str
    passed: bool
    details: str = ""
    error: str = ""


class RobustnessTester:
    """Stress-test the XGBoost model against adverse input conditions.

    Tests:
    1. missing_features — zero-fill each column in turn
    2. noisy_inputs — ±20% uniform noise on all continuous features
    3. low_data_team — sparse feature vector (most features zeroed)
    4. extreme_features — very large Elo gap

    Goal: no unhandled exceptions, probabilities remain in [0,1] and sum to ~1.0.
    """

    def __init__(self, xgb_model: XGBoostMatchModel, random_seed: int = 42) -> None:
        self._model = xgb_model
        self._rng = np.random.default_rng(random_seed)

    def run_all(self, base_features: dict[str, float]) -> list[StressTestResult]:
        results = [
            self._test_missing_features(base_features),
            self._test_noisy_inputs(base_features),
            self._test_low_data_team(base_features),
            self._test_extreme_features(base_features),
        ]
        passed = sum(1 for r in results if r.passed)
        logger.info(f"Stress tests: {passed}/{len(results)} passed")
        return results

    def to_markdown(self, results: list[StressTestResult]) -> str:
        lines = ["## Robustness Stress Test Results\n"]
        lines.append("| Test | Status | Details |")
        lines.append("|------|--------|---------|")
        for r in results:
            status = "✓ PASS" if r.passed else "✗ FAIL"
            detail = r.error if r.error else r.details
            lines.append(f"| {r.test_name} | {status} | {detail} |")
        return "\n".join(lines)

    def _predict_safe(self, features: dict[str, float]) -> StressTestResult | None:
        try:
            df = pd.DataFrame([features])
            results = self._model.predict_proba_dict(df)
            probs = results[0]
            total = sum(probs.values())
            for k, v in probs.items():
                if not (0.0 <= v <= 1.0):
                    return StressTestResult(
                        test_name="", passed=False,
                        error=f"Probability {k}={v:.4f} out of [0,1]"
                    )
            if abs(total - 1.0) > 0.01:
                return StressTestResult(
                    test_name="", passed=False,
                    error=f"Probabilities sum to {total:.4f}, expected ~1.0"
                )
            return None
        except Exception as exc:
            return StressTestResult(test_name="", passed=False, error=str(exc))

    def _test_missing_features(self, base: dict[str, float]) -> StressTestResult:
        name = "missing_features"
        failures = []
        for col in FEATURE_COLUMNS:
            zeroed = {**base, col: 0.0}
            err = self._predict_safe(zeroed)
            if err is not None:
                failures.append(f"{col}: {err.error}")
        if failures:
            return StressTestResult(test_name=name, passed=False,
                                    error=f"{len(failures)} columns caused failures")
        return StressTestResult(test_name=name, passed=True,
                                details=f"Zero-filled {len(FEATURE_COLUMNS)} columns — all valid")

    def _test_noisy_inputs(self, base: dict[str, float]) -> StressTestResult:
        name = "noisy_inputs"
        for trial in range(10):
            noisy = {}
            for col in FEATURE_COLUMNS:
                v = base.get(col, 0.0)
                if col == "is_neutral":
                    noisy[col] = v
                else:
                    noisy[col] = v * (1 + float(self._rng.uniform(-0.2, 0.2)))
            err = self._predict_safe(noisy)
            if err is not None:
                return StressTestResult(test_name=name, passed=False,
                                        error=f"Trial {trial}: {err.error}")
        return StressTestResult(test_name=name, passed=True,
                                details="10 noisy-input trials — all valid")

    def _test_low_data_team(self, base: dict[str, float]) -> StressTestResult:
        name = "low_data_team"
        sparse = {
            col: (base.get(col, 0.0) if col in ("home_elo", "away_elo", "elo_diff") else 0.0)
            for col in FEATURE_COLUMNS
        }
        err = self._predict_safe(sparse)
        if err:
            return StressTestResult(test_name=name, passed=False, error=err.error)
        return StressTestResult(test_name=name, passed=True,
                                details="Sparse feature vector — prediction valid")

    def _test_extreme_features(self, base: dict[str, float]) -> StressTestResult:
        name = "extreme_features"
        extreme = {**base, "elo_diff": 800.0, "home_elo": 2200.0, "away_elo": 1200.0}
        err = self._predict_safe(extreme)
        if err:
            return StressTestResult(test_name=name, passed=False, error=err.error)
        return StressTestResult(test_name=name, passed=True,
                                details="Extreme Elo gap (800 pts) — prediction valid and in [0,1]")
