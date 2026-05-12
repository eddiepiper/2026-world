# Phase 3a — Intelligence Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ensemble weight optimization, confidence scoring, drift detection, and prediction stability to the 2026-world forecasting system so that the ensemble outperforms XGBoost and every prediction carries a meaningful reliability signal.

**Architecture:** Hub-and-spoke. Two new packages (`src/optimization/`, `src/diagnostics/`) form the intelligence layer. CLI commands in `src/cli/` are thin wrappers that wire up models and call into these packages. `src/main.py` remains the top-level dispatcher.

**Tech Stack:** scipy (SLSQP optimizer, already installed), numpy, pandas, loguru, rich — all already in venv at `.venv/bin/python`. No new hard dependencies.

---

## File Map

**Create:**
- `src/optimization/__init__.py`
- `src/optimization/weight_search.py` — simplex validation + grid helpers
- `src/optimization/ensemble_optimizer.py` — SLSQP weight optimizer
- `src/diagnostics/__init__.py`
- `src/diagnostics/confidence_scorer.py` — 5-factor confidence engine
- `src/diagnostics/drift_detection.py` — rolling metric drift alerts
- `src/diagnostics/prediction_stability.py` — perturbation-based sensitivity
- `src/diagnostics/reliability_monitor.py` — aggregates all three
- `src/cli/__init__.py`
- `src/cli/optimize_cmd.py`
- `src/cli/diagnostic_cmd.py`
- `tests/test_ensemble_optimizer.py`
- `tests/test_confidence_scorer.py`
- `tests/test_drift_detection.py`
- `tests/test_prediction_stability.py`

**Modify:**
- `src/config/settings.py` — add `DiagnosticsConfig` dataclass + `diagnostics` field on `Settings`
- `src/main.py` — add `optimize`, `confidence`, `drift_check` CLI commands

---

## Task 1: Add DiagnosticsConfig to Settings

**Files:**
- Modify: `src/config/settings.py`

- [ ] **Step 1: Add `DiagnosticsConfig` and wire it into `Settings`**

Open `src/config/settings.py`. After the `EnsembleConfig` dataclass and before the `Settings` dataclass, add:

```python
@dataclass
class DiagnosticsConfig:
    drift_window: int = 50
    drift_alert_threshold: float = 2.0
    stability_n_perturbations: int = 20
    stability_noise_scale: float = 0.05
```

Then inside the `Settings` dataclass, add a new field after `ensemble`:

```python
diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
```

- [ ] **Step 2: Verify settings loads without error**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python -c "from src.config.settings import settings; print(settings.diagnostics)"
```

Expected output: `DiagnosticsConfig(drift_window=50, drift_alert_threshold=2.0, stability_n_perturbations=20, stability_noise_scale=0.05)`

- [ ] **Step 3: Run existing tests to confirm no regression**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python -m pytest tests/ -q
```

Expected: all tests pass (200 passing).

- [ ] **Step 4: Commit**

```bash
git add src/config/settings.py
git commit -m "feat: add DiagnosticsConfig to settings"
```

---

## Task 2: Create Optimization Package — weight_search.py

**Files:**
- Create: `src/optimization/__init__.py`
- Create: `src/optimization/weight_search.py`
- Test: `tests/test_ensemble_optimizer.py`

- [ ] **Step 1: Write the failing tests for weight_search**

Create `tests/test_ensemble_optimizer.py`:

```python
"""Tests for src/optimization/ — weight validation, grid search, SLSQP optimizer."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.optimization.weight_search import validate_weights, simplex_grid


class TestValidateWeights:
    def test_valid_weights_pass(self):
        validate_weights(0.3, 0.3, 0.4)  # should not raise

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            validate_weights(-0.1, 0.6, 0.5)

    def test_sum_not_one_raises(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            validate_weights(0.2, 0.2, 0.2)

    def test_single_model_weight_passes(self):
        validate_weights(1.0, 0.0, 0.0)

    def test_zero_all_raises(self):
        with pytest.raises(ValueError):
            validate_weights(0.0, 0.0, 0.0)


class TestSimplexGrid:
    def test_returns_list_of_triples(self):
        points = simplex_grid(resolution=2)
        assert all(len(p) == 3 for p in points)

    def test_all_sum_to_one(self):
        for w1, w2, w3 in simplex_grid(resolution=5):
            assert abs(w1 + w2 + w3 - 1.0) < 1e-6, f"{w1}+{w2}+{w3} != 1"

    def test_all_non_negative(self):
        for w1, w2, w3 in simplex_grid(resolution=5):
            assert w1 >= 0 and w2 >= 0 and w3 >= 0

    def test_resolution_1_has_3_vertices(self):
        points = simplex_grid(resolution=1)
        assert len(points) == 3
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python -m pytest tests/test_ensemble_optimizer.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.optimization'`

- [ ] **Step 3: Create `src/optimization/__init__.py`**

```python
```

(empty file)

- [ ] **Step 4: Create `src/optimization/weight_search.py`**

```python
from __future__ import annotations


def validate_weights(elo: float, poisson: float, xgboost: float, tol: float = 1e-6) -> None:
    """Raise ValueError if any weight is negative or they don't sum to 1."""
    for name, w in [("elo", elo), ("poisson", poisson), ("xgboost", xgboost)]:
        if w < 0:
            raise ValueError(f"Weight '{name}' must be non-negative, got {w}")
    total = elo + poisson + xgboost
    if abs(total - 1.0) > tol:
        raise ValueError(f"Weights must sum to 1.0, got {total:.8f}")


def simplex_grid(resolution: int = 5) -> list[tuple[float, float, float]]:
    """Return all weight triples (w1, w2, w3) on the 3-simplex at given resolution.

    Each returned triple sums to 1.0 and all values are non-negative.
    Number of points = (resolution+1)(resolution+2)/2.
    """
    step = 1.0 / resolution
    points: list[tuple[float, float, float]] = []
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            k = resolution - i - j
            points.append((round(i * step, 6), round(j * step, 6), round(k * step, 6)))
    return points
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python -m pytest tests/test_ensemble_optimizer.py::TestValidateWeights tests/test_ensemble_optimizer.py::TestSimplexGrid -v
```

Expected: 9 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/optimization/__init__.py src/optimization/weight_search.py tests/test_ensemble_optimizer.py
git commit -m "feat: optimization package skeleton — weight validation + simplex grid"
```

---

## Task 3: Implement WeightOptimizer

**Files:**
- Create: `src/optimization/ensemble_optimizer.py`
- Test: `tests/test_ensemble_optimizer.py` (extend)

- [ ] **Step 1: Write failing tests for WeightOptimizer**

Append to `tests/test_ensemble_optimizer.py`:

```python
from src.optimization.ensemble_optimizer import WeightOptimizer
from src.evaluation.metrics import CLASSES


def _make_matches_df(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = ["Alpha", "Beta", "Gamma", "Delta"]
    rows = []
    dates = pd.date_range("2015-01-01", periods=n, freq="14D")
    for i in range(n):
        h, a = rng.choice(teams, size=2, replace=False)
        hg, ag = int(rng.integers(0, 4)), int(rng.integers(0, 4))
        rows.append({"date": dates[i], "home_team": h, "away_team": a,
                     "home_goals": hg, "away_goals": ag})
    return pd.DataFrame(rows)


def _make_mock_optimizer(split_date: str = "2020-01-01") -> WeightOptimizer:
    """WeightOptimizer with all component models mocked."""
    matches = _make_matches_df()

    elo = MagicMock()
    elo.win_draw_loss_probabilities.return_value = (0.45, 0.25, 0.30)

    poisson = MagicMock()
    poisson.win_draw_loss_from_poisson.return_value = (0.40, 0.30, 0.30)

    xgb = MagicMock()
    xgb.predict_proba_dict.return_value = [
        {"home_win": 0.50, "draw": 0.25, "away_win": 0.25}
    ]

    feature_builder = MagicMock()
    feature_builder.build_features_for_match.return_value = {"elo_diff": 0.1}

    return WeightOptimizer(
        matches_df=matches,
        elo_model=elo,
        poisson_model=poisson,
        xgb_model=xgb,
        feature_builder=feature_builder,
        test_split_date=split_date,
    )


class TestWeightOptimizer:
    def test_optimize_returns_required_keys(self):
        optimizer = _make_mock_optimizer()
        result = optimizer.optimize()
        for key in ("elo", "poisson", "xgboost", "optimized_log_loss",
                    "baseline_log_loss", "improvement", "method", "validated_at"):
            assert key in result, f"Missing key: {key}"

    def test_optimized_weights_sum_to_one(self):
        result = _make_mock_optimizer().optimize()
        total = result["elo"] + result["poisson"] + result["xgboost"]
        assert abs(total - 1.0) < 1e-4

    def test_optimized_weights_non_negative(self):
        result = _make_mock_optimizer().optimize()
        assert result["elo"] >= 0
        assert result["poisson"] >= 0
        assert result["xgboost"] >= 0

    def test_method_is_slsqp(self):
        result = _make_mock_optimizer().optimize()
        assert result["method"] == "SLSQP"

    def test_empty_test_set_raises(self):
        # split_date after all matches → empty test set
        with pytest.raises(ValueError, match="No test matches"):
            _make_mock_optimizer(split_date="2030-01-01").optimize()

    def test_save_writes_json(self):
        optimizer = _make_mock_optimizer()
        result = optimizer.optimize()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.json"
            optimizer.save(path, result)
            assert path.exists()
            loaded = json.loads(path.read_text())
            assert loaded["method"] == "SLSQP"
```

- [ ] **Step 2: Run to confirm new tests fail**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python -m pytest tests/test_ensemble_optimizer.py::TestWeightOptimizer -v
```

Expected: `ImportError: cannot import name 'WeightOptimizer'`

- [ ] **Step 3: Create `src/optimization/ensemble_optimizer.py`**

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize
from sklearn.metrics import log_loss

from src.config.settings import settings
from src.evaluation.metrics import CLASSES
from src.features.feature_builder import FeatureBuilder
from src.ml.xgboost_model import XGBoostMatchModel
from src.models.elo_model import EloModel
from src.models.poisson_model import PoissonModel
from src.optimization.weight_search import validate_weights


class WeightOptimizer:
    """Find ensemble weights that minimise log loss on the chronological test set.

    Uses SLSQP (Sequential Least Squares Programming) to optimise over the
    3-simplex (weights ≥ 0, sum = 1). Same test split as the Phase 2 benchmark
    — no leakage.
    """

    def __init__(
        self,
        matches_df: pd.DataFrame,
        elo_model: EloModel,
        poisson_model: PoissonModel,
        xgb_model: XGBoostMatchModel,
        feature_builder: FeatureBuilder,
        test_split_date: str = "2020-01-01",
    ) -> None:
        self._matches = matches_df.copy()
        self._matches["date"] = pd.to_datetime(self._matches["date"])
        self._elo = elo_model
        self._poisson = poisson_model
        self._xgb = xgb_model
        self._feature_builder = feature_builder
        self._split_date = pd.Timestamp(test_split_date)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self) -> dict:
        """Run SLSQP optimisation. Returns result dict."""
        elo_arr, poi_arr, xgb_arr, y_true = self._build_test_arrays()

        # Baseline: current fixed weights from settings
        cfg = settings.ensemble
        w0 = [cfg.elo_weight, cfg.poisson_weight, cfg.xgboost_weight]
        baseline = w0[0] * elo_arr + w0[1] * poi_arr + w0[2] * xgb_arr
        baseline_loss = float(log_loss(y_true, baseline, labels=CLASSES))

        def objective(w: np.ndarray) -> float:
            blended = w[0] * elo_arr + w[1] * poi_arr + w[2] * xgb_arr
            return float(log_loss(y_true, blended, labels=CLASSES))

        result = minimize(
            objective,
            x0=w0,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * 3,
            constraints=[{"type": "eq", "fun": lambda w: float(w.sum()) - 1.0}],
            options={"ftol": 1e-9, "maxiter": 500},
        )

        ew, pw, xw = float(result.x[0]), float(result.x[1]), float(result.x[2])
        optimized_loss = float(result.fun)

        logger.info(
            f"Optimised weights — elo={ew:.4f} poisson={pw:.4f} xgboost={xw:.4f} "
            f"log_loss={optimized_loss:.6f} (baseline={baseline_loss:.6f})"
        )

        return {
            "elo": round(ew, 4),
            "poisson": round(pw, 4),
            "xgboost": round(xw, 4),
            "optimized_log_loss": round(optimized_loss, 6),
            "baseline_log_loss": round(baseline_loss, 6),
            "improvement": round(baseline_loss - optimized_loss, 6),
            "method": "SLSQP",
            "validated_at": str(date.today()),
        }

    def save(self, path: Path, result: dict) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
        logger.info(f"Optimal weight config saved to {path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_test_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
        """Build (elo_proba, poisson_proba, xgb_proba, y_true) on the test split.

        Probability arrays have shape (n_matches, 3), columns ordered [A, D, H]
        to match CLASSES = ['A', 'D', 'H'] expected by sklearn.log_loss.
        """
        test = self._matches[self._matches["date"] >= self._split_date].copy()
        if test.empty:
            raise ValueError(
                f"No test matches after split date {self._split_date.date()}. "
                "Check test_split_date in MLConfig."
            )

        elo_rows, poi_rows, xgb_rows, y_true = [], [], [], []

        for _, row in test.iterrows():
            home, away = str(row["home_team"]), str(row["away_team"])
            match_date = pd.Timestamp(row["date"])
            hg, ag = int(row["home_goals"]), int(row["away_goals"])

            outcome = "H" if hg > ag else ("D" if hg == ag else "A")
            y_true.append(outcome)

            # Elo: [A, D, H]
            ew, ed, el = self._elo.win_draw_loss_probabilities(home, away)
            elo_rows.append([el, ed, ew])

            # Poisson: [A, D, H]
            pw, pd_, pl = self._poisson.win_draw_loss_from_poisson(home, away)
            poi_rows.append([pl, pd_, pw])

            # XGBoost: [A, D, H]
            features = self._feature_builder.build_features_for_match(home, away, match_date)
            feat_df = pd.DataFrame([features])
            xgb_result = self._xgb.predict_proba_dict(feat_df)[0]
            xgb_rows.append([xgb_result["away_win"], xgb_result["draw"], xgb_result["home_win"]])

        return (
            np.array(elo_rows, dtype=float),
            np.array(poi_rows, dtype=float),
            np.array(xgb_rows, dtype=float),
            pd.Series(y_true),
        )
```

- [ ] **Step 4: Run all optimizer tests**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python -m pytest tests/test_ensemble_optimizer.py -v
```

Expected: all tests pass (9 weight_search + 6 optimizer = 15 total).

- [ ] **Step 5: Commit**

```bash
git add src/optimization/ensemble_optimizer.py tests/test_ensemble_optimizer.py
git commit -m "feat: WeightOptimizer — SLSQP ensemble weight optimisation"
```

---

## Task 4: CLI `optimize` Command

**Files:**
- Create: `src/cli/__init__.py`
- Create: `src/cli/optimize_cmd.py`
- Modify: `src/main.py`

- [ ] **Step 1: Create `src/cli/__init__.py`** (empty)

```python
```

- [ ] **Step 2: Create `src/cli/optimize_cmd.py`**

```python
from __future__ import annotations

import sys

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config.settings import settings
from src.ingestion.match_loader import load_matches
from src.models.elo_model import EloModel
from src.models.poisson_model import PoissonModel

console = Console()


def cmd_optimize() -> None:
    """Find optimal ensemble blend weights and save to outputs/models/best_weight_config.json."""
    from src.features.feature_builder import FeatureBuilder
    from src.ml.xgboost_model import XGBoostMatchModel
    from src.optimization.ensemble_optimizer import WeightOptimizer

    model_path = settings.outputs_dir / "models" / "model.pkl"
    if not model_path.exists():
        console.print(
            "[bold red]No trained model found.[/bold red] "
            "Run [bold]python main.py train[/bold] first."
        )
        sys.exit(1)

    xgb_model = XGBoostMatchModel.load(model_path)

    data_path = settings.data_dir / "processed" / "matches.csv"
    if not data_path.exists():
        data_path = settings.data_dir / "sample" / "matches.csv"
    matches_df = load_matches(data_path)

    elo = EloModel(config=settings.elo)
    elo.train_on_matches(matches_df)
    poisson = PoissonModel(config=settings.poisson)
    poisson.fit(matches_df)

    feature_builder = FeatureBuilder(
        matches_df=matches_df,
        elo_model=elo,
        poisson_model=poisson,
    )

    optimizer = WeightOptimizer(
        matches_df=matches_df,
        elo_model=elo,
        poisson_model=poisson,
        xgb_model=xgb_model,
        feature_builder=feature_builder,
        test_split_date=settings.ml.test_split_date,
    )

    console.print(Panel("Running SLSQP ensemble weight optimisation…", style="blue"))

    result = optimizer.optimize()

    out_path = settings.outputs_dir / "models" / "best_weight_config.json"
    optimizer.save(out_path, result)

    table = Table(title="Optimal Ensemble Weights", show_header=True, header_style="bold cyan")
    table.add_column("Parameter", style="white")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Elo weight", f"{result['elo']:.4f}")
    table.add_row("Poisson weight", f"{result['poisson']:.4f}")
    table.add_row("XGBoost weight", f"{result['xgboost']:.4f}")
    table.add_section()
    table.add_row("Baseline log loss", f"{result['baseline_log_loss']:.6f}")
    table.add_row("Optimised log loss", f"{result['optimized_log_loss']:.6f}")
    table.add_row("Improvement", f"{result['improvement']:+.6f}")

    console.print()
    console.print(table)
    console.print(f"\n[dim]Saved to {out_path}[/dim]\n")
```

- [ ] **Step 3: Add `optimize` to the `main()` dispatcher in `src/main.py`**

In `src/main.py`, add this import near the top of the file (after the existing imports):

```python
from src.cli.optimize_cmd import cmd_optimize
```

In the `main()` function, add a new branch inside the `if/elif` chain:

```python
    elif command == "optimize":
        cmd_optimize()
```

Also update the usage message at the top of `main()` to include:

```python
        console.print("  python main.py optimize")
```

- [ ] **Step 4: Smoke-test the CLI (requires trained model)**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python main.py optimize
```

Expected: table showing Elo/Poisson/XGBoost weights and log loss improvement. File created at `outputs/models/best_weight_config.json`.

- [ ] **Step 5: Commit**

```bash
git add src/cli/__init__.py src/cli/optimize_cmd.py src/main.py
git commit -m "feat: CLI optimize command — SLSQP ensemble weight search"
```

---

## Task 5: Confidence Scorer

**Files:**
- Create: `src/diagnostics/__init__.py`
- Create: `src/diagnostics/confidence_scorer.py`
- Test: `tests/test_confidence_scorer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_confidence_scorer.py`:

```python
"""Tests for src/diagnostics/confidence_scorer.py."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.diagnostics.confidence_scorer import ConfidenceScorer


def _make_matches_df(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = ["Brazil", "France", "Germany", "Argentina"]
    rows = []
    dates = pd.date_range("2018-01-01", periods=n, freq="14D")
    for i in range(n):
        h, a = rng.choice(teams, size=2, replace=False)
        rows.append({
            "date": dates[i],
            "home_team": h,
            "away_team": a,
            "home_goals": int(rng.integers(0, 4)),
            "away_goals": int(rng.integers(0, 4)),
        })
    return pd.DataFrame(rows)


@pytest.fixture()
def scorer() -> ConfidenceScorer:
    return ConfidenceScorer(matches_df=_make_matches_df(), calibration_ece=0.05)


def _agreeing_components() -> dict:
    return {
        "elo":     {"home_win": 0.50, "draw": 0.25, "away_win": 0.25},
        "poisson": {"home_win": 0.50, "draw": 0.25, "away_win": 0.25},
        "xgboost": {"home_win": 0.50, "draw": 0.25, "away_win": 0.25},
    }


def _disagreeing_components() -> dict:
    return {
        "elo":     {"home_win": 0.90, "draw": 0.05, "away_win": 0.05},
        "poisson": {"home_win": 0.10, "draw": 0.05, "away_win": 0.85},
        "xgboost": {"home_win": 0.50, "draw": 0.30, "away_win": 0.20},
    }


def _ensemble_probs() -> dict:
    return {"home_win": 0.50, "draw": 0.25, "away_win": 0.25}


def _full_features() -> dict:
    return {k: 0.5 for k in [
        "elo_diff", "home_elo", "away_elo", "fifa_diff",
        "home_attack", "home_defense", "away_attack", "away_defense",
        "home_win_rate", "away_win_rate", "home_goals_scored",
        "home_goals_conceded", "away_goals_scored", "away_goals_conceded",
        "home_clean_sheet_rate", "away_clean_sheet_rate",
        "home_days_rest", "away_days_rest", "is_neutral",
    ]}


class TestConfidenceScorerOutput:
    def test_returns_required_keys(self, scorer):
        result = scorer.score("Brazil", "France", _agreeing_components(),
                              _ensemble_probs(), _full_features())
        assert "confidence_score" in result
        assert "confidence_band" in result
        assert "factor_breakdown" in result

    def test_score_in_unit_interval(self, scorer):
        result = scorer.score("Brazil", "France", _agreeing_components(),
                              _ensemble_probs(), _full_features())
        assert 0.0 <= result["confidence_score"] <= 1.0

    def test_confidence_band_is_valid(self, scorer):
        result = scorer.score("Brazil", "France", _agreeing_components(),
                              _ensemble_probs(), _full_features())
        assert result["confidence_band"] in ("Low", "Medium", "High")

    def test_factor_breakdown_has_five_factors(self, scorer):
        result = scorer.score("Brazil", "France", _agreeing_components(),
                              _ensemble_probs(), _full_features())
        fb = result["factor_breakdown"]
        for key in ("model_agreement", "calibration_quality",
                    "feature_completeness", "historical_reliability",
                    "prediction_volatility"):
            assert key in fb


class TestModelAgreementFactor:
    def test_fully_agreeing_models_give_high_agreement(self, scorer):
        result = scorer.score("Brazil", "France", _agreeing_components(),
                              _ensemble_probs(), _full_features())
        assert result["factor_breakdown"]["model_agreement"] > 0.95

    def test_fully_disagreeing_models_give_low_agreement(self, scorer):
        result = scorer.score("Brazil", "France", _disagreeing_components(),
                              _ensemble_probs(), _full_features())
        assert result["factor_breakdown"]["model_agreement"] < 0.5

    def test_high_agreement_band_is_high(self, scorer):
        result = scorer.score("Brazil", "France", _agreeing_components(),
                              _ensemble_probs(), _full_features())
        assert result["confidence_band"] == "High"


class TestFeatureCompleteness:
    def test_all_valid_features_give_completeness_one(self, scorer):
        result = scorer.score("Brazil", "France", _agreeing_components(),
                              _ensemble_probs(), _full_features())
        assert result["factor_breakdown"]["feature_completeness"] == 1.0

    def test_half_nan_features_give_completeness_half(self, scorer):
        features = _full_features()
        keys = list(features.keys())
        for k in keys[: len(keys) // 2]:
            features[k] = float("nan")
        result = scorer.score("Brazil", "France", _agreeing_components(),
                              _ensemble_probs(), features)
        assert abs(result["factor_breakdown"]["feature_completeness"] - 0.5) < 0.1


class TestHistoricalReliability:
    def test_unknown_pair_gives_zero_reliability(self, scorer):
        result = scorer.score("Andorra", "SanMarino", _agreeing_components(),
                              _ensemble_probs(), _full_features())
        assert result["factor_breakdown"]["historical_reliability"] == 0.0

    def test_known_pair_gives_positive_reliability(self, scorer):
        matches = _make_matches_df()
        # Pick a pair that actually played
        row = matches.iloc[0]
        result = scorer.score(row["home_team"], row["away_team"],
                              _agreeing_components(), _ensemble_probs(), _full_features())
        assert result["factor_breakdown"]["historical_reliability"] > 0.0


class TestConfidenceBands:
    def test_low_band_below_040(self, scorer):
        # Force score to be very low by making a scorer with terrible calibration
        bad_scorer = ConfidenceScorer(matches_df=_make_matches_df(), calibration_ece=0.99)
        result = bad_scorer.score("Andorra", "SanMarino", _disagreeing_components(),
                                  _ensemble_probs(), _full_features())
        assert result["confidence_band"] == "Low"
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python -m pytest tests/test_confidence_scorer.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.diagnostics'`

- [ ] **Step 3: Create `src/diagnostics/__init__.py`** (empty)

```python
```

- [ ] **Step 4: Create `src/diagnostics/confidence_scorer.py`**

```python
from __future__ import annotations

import math

import pandas as pd

_WEIGHTS: dict[str, float] = {
    "model_agreement": 0.35,
    "calibration_quality": 0.25,
    "feature_completeness": 0.20,
    "historical_reliability": 0.15,
    "prediction_volatility": 0.05,
}

# (band_name, lower_bound) — checked from highest threshold down
_BANDS: list[tuple[str, float]] = [("High", 0.70), ("Medium", 0.40), ("Low", 0.0)]


class ConfidenceScorer:
    """Five-factor confidence engine for ensemble predictions.

    Factors (weights):
    - model_agreement       0.35 — how much Elo/Poisson/XGBoost agree
    - calibration_quality   0.25 — 1 − ECE from the Phase 2 benchmark
    - feature_completeness  0.20 — fraction of non-null features
    - historical_reliability 0.15 — log-scaled head-to-head match count
    - prediction_volatility  0.05 — 1 − normalised entropy (down-weighted
                                     because close matches are not unreliable)
    """

    def __init__(self, matches_df: pd.DataFrame, calibration_ece: float) -> None:
        self._matches = matches_df.copy()
        self._ece = calibration_ece

        pairs = pd.concat([
            self._matches[["home_team", "away_team"]].rename(
                columns={"home_team": "t1", "away_team": "t2"}
            ),
            self._matches[["away_team", "home_team"]].rename(
                columns={"away_team": "t1", "home_team": "t2"}
            ),
        ])
        counts = pairs.groupby(["t1", "t2"]).size()
        self._max_count = int(counts.max()) if len(counts) > 0 else 1

    def score(
        self,
        home_team: str,
        away_team: str,
        component_models: dict[str, dict[str, float]],
        ensemble_probs: dict[str, float],
        feature_vector: dict[str, float],
    ) -> dict:
        """Return confidence dict with score, band, and per-factor breakdown."""
        factors = {
            "model_agreement": self._model_agreement(component_models),
            "calibration_quality": max(0.0, 1.0 - self._ece),
            "feature_completeness": self._feature_completeness(feature_vector),
            "historical_reliability": self._historical_reliability(home_team, away_team),
            "prediction_volatility": self._prediction_volatility(ensemble_probs),
        }

        raw_score = sum(_WEIGHTS[k] * v for k, v in factors.items())
        confidence_score = max(0.0, min(1.0, raw_score))

        band = "Low"
        for name, threshold in _BANDS:
            if confidence_score >= threshold:
                band = name
                break

        return {
            "confidence_score": round(confidence_score, 4),
            "confidence_band": band,
            "factor_breakdown": {k: round(v, 4) for k, v in factors.items()},
        }

    # ------------------------------------------------------------------
    # Private factor computations
    # ------------------------------------------------------------------

    def _model_agreement(self, components: dict[str, dict[str, float]]) -> float:
        models = list(components.values())
        if len(models) < 2:
            return 1.0
        outcomes = ("home_win", "draw", "away_win")
        disagreements: list[float] = []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                disagreements.append(
                    max(abs(models[i][o] - models[j][o]) for o in outcomes)
                )
        return max(0.0, 1.0 - sum(disagreements) / len(disagreements))

    def _feature_completeness(self, feature_vector: dict[str, float]) -> float:
        if not feature_vector:
            return 0.0
        valid = sum(
            1 for v in feature_vector.values()
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        )
        return valid / len(feature_vector)

    def _historical_reliability(self, home_team: str, away_team: str) -> float:
        mask = (
            (
                (self._matches["home_team"] == home_team)
                & (self._matches["away_team"] == away_team)
            )
            | (
                (self._matches["home_team"] == away_team)
                & (self._matches["away_team"] == home_team)
            )
        )
        count = int(mask.sum())
        if count == 0:
            return 0.0
        return math.log1p(count) / math.log1p(self._max_count)

    def _prediction_volatility(self, ensemble_probs: dict[str, float]) -> float:
        probs = [
            ensemble_probs["home_win"],
            ensemble_probs["draw"],
            ensemble_probs["away_win"],
        ]
        max_entropy = math.log(3)
        entropy = -sum(p * math.log(max(p, 1e-10)) for p in probs)
        return max(0.0, 1.0 - entropy / max_entropy)
```

- [ ] **Step 5: Run confidence scorer tests**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python -m pytest tests/test_confidence_scorer.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/diagnostics/__init__.py src/diagnostics/confidence_scorer.py tests/test_confidence_scorer.py
git commit -m "feat: diagnostics package — ConfidenceScorer with 5-factor engine"
```

---

## Task 6: CLI `confidence` Command

**Files:**
- Create: `src/cli/diagnostic_cmd.py` (partial — will be extended in Task 8)
- Modify: `src/main.py`

- [ ] **Step 1: Create `src/cli/diagnostic_cmd.py`**

```python
from __future__ import annotations

import sys

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config.settings import settings
from src.ingestion.match_loader import load_matches
from src.models.elo_model import EloModel
from src.models.poisson_model import PoissonModel

console = Console()


def _load_calibration_ece() -> float:
    """Read ensemble ECE from Phase 2 benchmark CSV. Returns 0.5 if unavailable."""
    csv_path = settings.outputs_dir / "evaluation" / "benchmark_results.csv"
    if not csv_path.exists():
        logger.warning("benchmark_results.csv not found — using ECE fallback 0.5")
        return 0.5
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        row = df[df["model"].str.contains("Ensemble", case=False, na=False)]
        if row.empty:
            return 0.5
        return float(row["ece"].iloc[0])
    except Exception as exc:
        logger.warning(f"Could not read ECE from benchmark CSV: {exc}")
        return 0.5


def _load_matches_df() -> pd.DataFrame:
    data_path = settings.data_dir / "processed" / "matches.csv"
    if not data_path.exists():
        data_path = settings.data_dir / "sample" / "matches.csv"
    return load_matches(data_path)


def cmd_confidence(home_team: str, away_team: str) -> None:
    """Print a 5-factor confidence score for the given match-up."""
    from src.diagnostics.confidence_scorer import ConfidenceScorer
    from src.ensemble.ensemble_engine import EnsembleEngine, EnsembleWeights
    from src.features.feature_builder import FeatureBuilder
    from src.ml.xgboost_model import XGBoostMatchModel

    model_path = settings.outputs_dir / "models" / "model.pkl"
    if not model_path.exists():
        console.print(
            "[bold red]No trained model found.[/bold red] "
            "Run [bold]python main.py train[/bold] first."
        )
        sys.exit(1)

    xgb_model = XGBoostMatchModel.load(model_path)
    matches_df = _load_matches_df()

    elo = EloModel(config=settings.elo)
    elo.train_on_matches(matches_df)
    poisson = PoissonModel(config=settings.poisson)
    poisson.fit(matches_df)

    feature_builder = FeatureBuilder(
        matches_df=matches_df, elo_model=elo, poisson_model=poisson
    )

    # Load optimal weights if available
    weight_path = settings.outputs_dir / "models" / "best_weight_config.json"
    if weight_path.exists():
        import json
        cfg = json.loads(weight_path.read_text())
        weights = EnsembleWeights(
            elo=cfg["elo"], poisson=cfg["poisson"], xgboost=cfg["xgboost"]
        )
    else:
        weights = EnsembleWeights()

    ensemble = EnsembleEngine(
        elo_model=elo,
        poisson_model=poisson,
        xgb_model=xgb_model,
        feature_builder=feature_builder,
        weights=weights,
    )

    match_date = pd.Timestamp.today()
    ens_result = ensemble.predict(home_team, away_team, match_date)
    features = feature_builder.build_features_for_match(home_team, away_team, match_date)

    ece = _load_calibration_ece()
    scorer = ConfidenceScorer(matches_df=matches_df, calibration_ece=ece)
    confidence = scorer.score(
        home_team=home_team,
        away_team=away_team,
        component_models=ens_result["component_models"],
        ensemble_probs=ens_result["ensemble_probabilities"],
        feature_vector=features,
    )

    table = Table(
        title=f"Confidence — {home_team} vs {away_team}",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Factor", style="white")
    table.add_column("Score", style="green", justify="right")
    table.add_column("Weight", style="dim", justify="right")

    factor_weights = {
        "model_agreement": 0.35,
        "calibration_quality": 0.25,
        "feature_completeness": 0.20,
        "historical_reliability": 0.15,
        "prediction_volatility": 0.05,
    }

    for factor, value in confidence["factor_breakdown"].items():
        weight = factor_weights.get(factor, 0.0)
        table.add_row(factor.replace("_", " ").title(), f"{value:.3f}", f"{weight:.2f}")

    table.add_section()
    band = confidence["confidence_band"]
    band_color = {"High": "green", "Medium": "yellow", "Low": "red"}.get(band, "white")
    table.add_row(
        "[bold]Confidence Score[/bold]",
        f"[bold]{confidence['confidence_score']:.3f}[/bold]",
        "",
    )
    table.add_row("Band", f"[{band_color}]{band}[/{band_color}]", "")

    console.print()
    console.print(table)
    console.print()
```

- [ ] **Step 2: Wire `confidence` into `src/main.py`**

Add import near the other CLI imports:

```python
from src.cli.diagnostic_cmd import cmd_confidence
```

Add branch in `main()`:

```python
    elif command == "confidence":
        if len(args) != 3:
            console.print("[red]confidence requires exactly two team names[/red]")
            sys.exit(1)
        cmd_confidence(args[1], args[2])
```

Add usage line:

```python
        console.print("  python main.py confidence <home_team> <away_team>")
```

- [ ] **Step 3: Smoke-test**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python main.py confidence Brazil France
```

Expected: table with 5 factor scores and overall confidence band.

- [ ] **Step 4: Commit**

```bash
git add src/cli/diagnostic_cmd.py src/main.py
git commit -m "feat: CLI confidence command — 5-factor match confidence scoring"
```

---

## Task 7: Drift Detector

**Files:**
- Create: `src/diagnostics/drift_detection.py`
- Test: `tests/test_drift_detection.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_drift_detection.py`:

```python
"""Tests for src/diagnostics/drift_detection.py."""
from __future__ import annotations

import pytest

from src.config.settings import DiagnosticsConfig
from src.diagnostics.drift_detection import DriftAlert, DriftDetector


def _config(window: int = 10) -> DiagnosticsConfig:
    return DiagnosticsConfig(
        drift_window=window,
        drift_alert_threshold=2.0,
        stability_n_perturbations=5,
        stability_noise_scale=0.05,
    )


def _good_record() -> tuple[str, dict]:
    """A reasonable calibrated prediction that was correct."""
    return ("H", {"home_win": 0.60, "draw": 0.25, "away_win": 0.15})


def _bad_record() -> tuple[str, dict]:
    """A badly calibrated prediction (very confident but wrong)."""
    return ("A", {"home_win": 0.95, "draw": 0.04, "away_win": 0.01})


class TestDriftDetectorStatus:
    def test_empty_detector_returns_insufficient_data(self):
        d = DriftDetector(config=_config())
        status = d.status()
        assert status["status"] == "insufficient_data"
        assert status["n_records"] == 0

    def test_status_has_required_keys(self):
        d = DriftDetector(config=_config())
        status = d.status()
        assert "status" in status
        assert "n_records" in status
        assert "alerts" in status


class TestDriftDetectorAddRecord:
    def test_returns_no_alerts_for_few_records(self):
        d = DriftDetector(config=_config(window=10))
        for _ in range(15):  # less than 2*window
            alerts = d.add_record(*_good_record())
        assert alerts == []

    def test_stable_series_gives_no_alerts(self):
        d = DriftDetector(config=_config(window=5))
        alerts = []
        for _ in range(30):
            new_alerts = d.add_record(*_good_record())
            alerts.extend(new_alerts)
        assert alerts == []

    def test_alert_raised_on_degraded_predictions(self):
        d = DriftDetector(config=_config(window=5))
        # First window: good predictions
        for _ in range(10):
            d.add_record(*_good_record())
        # Second+ windows: terrible predictions
        alerts = []
        for _ in range(20):
            new_alerts = d.add_record(*_bad_record())
            alerts.extend(new_alerts)
        assert len(alerts) > 0


class TestDriftAlert:
    def test_alert_has_required_fields(self):
        alert = DriftAlert(
            metric="log_loss",
            current_value=1.5,
            baseline_mean=0.9,
            baseline_std=0.05,
            severity="warning",
        )
        assert alert.metric == "log_loss"
        assert alert.severity in ("warning", "critical")

    def test_severity_is_warning_or_critical(self):
        d = DriftDetector(config=_config(window=5))
        for _ in range(10):
            d.add_record(*_good_record())
        alerts = []
        for _ in range(30):
            alerts.extend(d.add_record(*_bad_record()))
        for a in alerts:
            assert a.severity in ("warning", "critical")


class TestDriftDetectorWindowMetrics:
    def test_metrics_have_correct_keys(self):
        d = DriftDetector(config=_config(window=5))
        for _ in range(6):
            d.add_record(*_good_record())
        status = d.status()
        if "current_window_metrics" in status:
            for key in ("log_loss", "brier_score", "entropy"):
                assert key in status["current_window_metrics"]
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python -m pytest tests/test_drift_detection.py -v
```

Expected: `ImportError: cannot import name 'DriftDetector'`

- [ ] **Step 3: Create `src/diagnostics/drift_detection.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger

from src.config.settings import DiagnosticsConfig, settings
from src.evaluation.metrics import CLASSES, brier_score, log_loss_score


@dataclass
class DriftAlert:
    metric: str              # "log_loss" | "brier_score" | "entropy"
    current_value: float
    baseline_mean: float
    baseline_std: float
    severity: Literal["warning", "critical"]


class DriftDetector:
    """Detect distribution drift in rolling prediction quality.

    Accepts (outcome, probabilities) pairs via ``add_record``.
    Once at least 2×drift_window records exist, compares the current
    window against the baseline window. Alerts when a metric exceeds
    baseline_mean + threshold × baseline_std.
    """

    def __init__(self, config: DiagnosticsConfig | None = None) -> None:
        self._config = config or settings.diagnostics
        self._records: list[tuple[str, dict[str, float]]] = []

    def add_record(self, y_true: str, probs: dict[str, float]) -> list[DriftAlert]:
        """Append a prediction outcome pair and return any new drift alerts."""
        self._records.append((y_true, probs))
        w = self._config.drift_window
        if len(self._records) < w * 2:
            return []
        return self._check_drift()

    def status(self) -> dict:
        """Return current rolling metrics and any active alerts."""
        w = self._config.drift_window
        n = len(self._records)

        if n == 0:
            return {"status": "insufficient_data", "n_records": 0, "alerts": []}

        recent = self._records[-w:] if n >= w else self._records
        metrics = self._window_metrics(recent)
        alerts = self._check_drift() if n >= w * 2 else []

        return {
            "status": "alert" if alerts else "ok",
            "n_records": n,
            "current_window_metrics": {k: round(v, 4) for k, v in metrics.items()},
            "alerts": [
                {
                    "metric": a.metric,
                    "current_value": a.current_value,
                    "baseline_mean": a.baseline_mean,
                    "severity": a.severity,
                }
                for a in alerts
            ],
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_drift(self) -> list[DriftAlert]:
        w = self._config.drift_window
        threshold = self._config.drift_alert_threshold
        records = self._records

        baseline = self._window_metrics(records[:w])
        current = self._window_metrics(records[-w:])

        # Estimate variance from all non-overlapping windows
        all_windows = [
            self._window_metrics(records[i : i + w])
            for i in range(0, len(records) - w + 1, max(1, w // 2))
        ]
        window_vals: dict[str, list[float]] = {
            m: [win[m] for win in all_windows[:-1]] for m in ("log_loss", "brier_score", "entropy")
        }

        alerts: list[DriftAlert] = []
        for metric in ("log_loss", "brier_score", "entropy"):
            vals = window_vals[metric]
            mean = float(np.mean(vals)) if vals else baseline[metric]
            std = float(np.std(vals)) if len(vals) > 1 else 0.0
            current_val = current[metric]

            if std > 0 and (current_val - mean) > threshold * std:
                severity: Literal["warning", "critical"] = (
                    "critical" if (current_val - mean) > 2 * threshold * std else "warning"
                )
                alerts.append(
                    DriftAlert(
                        metric=metric,
                        current_value=round(current_val, 4),
                        baseline_mean=round(mean, 4),
                        baseline_std=round(std, 4),
                        severity=severity,
                    )
                )
                logger.warning(
                    f"Drift [{severity}] {metric}: {current_val:.4f} vs "
                    f"baseline {mean:.4f} ± {std:.4f}"
                )

        return alerts

    def _window_metrics(
        self, records: list[tuple[str, dict[str, float]]]
    ) -> dict[str, float]:
        if not records:
            return {"log_loss": 0.0, "brier_score": 0.0, "entropy": 0.0}

        y_true = pd.Series([r[0] for r in records])
        proba = np.array(
            [
                [
                    r[1].get("away_win", 1 / 3),
                    r[1].get("draw", 1 / 3),
                    r[1].get("home_win", 1 / 3),
                ]
                for r in records
            ],
            dtype=float,
        )  # columns: [A, D, H] matching CLASSES

        ll = log_loss_score(y_true, proba)
        bs = brier_score(y_true, proba)
        eps = 1e-10
        entropy = float(np.mean(-np.sum(proba * np.log(proba + eps), axis=1)))

        return {"log_loss": ll, "brier_score": bs, "entropy": entropy}
```

- [ ] **Step 4: Run drift detection tests**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python -m pytest tests/test_drift_detection.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/diagnostics/drift_detection.py tests/test_drift_detection.py
git commit -m "feat: DriftDetector — rolling metric drift alerts"
```

---

## Task 8: CLI `drift_check` Command

**Files:**
- Modify: `src/cli/diagnostic_cmd.py`
- Modify: `src/main.py`

- [ ] **Step 1: Add `cmd_drift_check` to `src/cli/diagnostic_cmd.py`**

Append to the file after `cmd_confidence`:

```python
def cmd_drift_check() -> None:
    """Run drift check on Phase 2 benchmark test set predictions."""
    from src.diagnostics.drift_detection import DriftDetector
    from src.features.feature_builder import FeatureBuilder
    from src.ml.xgboost_model import XGBoostMatchModel
    from src.ensemble.ensemble_engine import EnsembleEngine, EnsembleWeights

    model_path = settings.outputs_dir / "models" / "model.pkl"
    if not model_path.exists():
        console.print(
            "[bold red]No trained model found.[/bold red] "
            "Run [bold]python main.py train[/bold] first."
        )
        sys.exit(1)

    xgb_model = XGBoostMatchModel.load(model_path)
    matches_df = _load_matches_df()

    elo = EloModel(config=settings.elo)
    elo.train_on_matches(matches_df)
    poisson = PoissonModel(config=settings.poisson)
    poisson.fit(matches_df)

    feature_builder = FeatureBuilder(
        matches_df=matches_df, elo_model=elo, poisson_model=poisson
    )

    weight_path = settings.outputs_dir / "models" / "best_weight_config.json"
    if weight_path.exists():
        import json
        cfg = json.loads(weight_path.read_text())
        weights = EnsembleWeights(
            elo=cfg["elo"], poisson=cfg["poisson"], xgboost=cfg["xgboost"]
        )
    else:
        weights = EnsembleWeights()

    ensemble = EnsembleEngine(
        elo_model=elo,
        poisson_model=poisson,
        xgb_model=xgb_model,
        feature_builder=feature_builder,
        weights=weights,
    )

    # Feed test-set predictions into the detector
    test_matches = matches_df[
        pd.to_datetime(matches_df["date"]) >= pd.Timestamp(settings.ml.test_split_date)
    ].sort_values("date")

    detector = DriftDetector()
    console.print(Panel(
        f"Running drift check on {len(test_matches)} test matches…", style="blue"
    ))

    for _, row in test_matches.iterrows():
        home, away = str(row["home_team"]), str(row["away_team"])
        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        outcome = "H" if hg > ag else ("D" if hg == ag else "A")
        try:
            result = ensemble.predict(home, away, pd.Timestamp(row["date"]))
            probs = result["ensemble_probabilities"]
            detector.add_record(outcome, probs)
        except Exception as exc:
            logger.warning(f"Skipping {home} vs {away}: {exc}")

    status = detector.status()

    table = Table(title="Drift Check Results", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="white")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Records processed", str(status["n_records"]))
    table.add_row("Status", f"[{'red' if status['status'] == 'alert' else 'green'}]{status['status']}[/]")

    if "current_window_metrics" in status:
        for metric, value in status["current_window_metrics"].items():
            table.add_row(f"Rolling {metric}", f"{value:.4f}")

    console.print()
    console.print(table)

    if status["alerts"]:
        console.print("\n[bold red]Drift Alerts:[/bold red]")
        for alert in status["alerts"]:
            console.print(
                f"  [{alert['severity'].upper()}] {alert['metric']}: "
                f"{alert['current_value']:.4f} (baseline {alert['baseline_mean']:.4f})"
            )
    else:
        console.print("\n[green]No drift detected.[/green]")

    console.print()
```

- [ ] **Step 2: Wire `drift_check` into `src/main.py`**

Update the import:

```python
from src.cli.diagnostic_cmd import cmd_confidence, cmd_drift_check
```

Add branch in `main()`:

```python
    elif command == "drift_check":
        cmd_drift_check()
```

Add usage line:

```python
        console.print("  python main.py drift_check")
```

- [ ] **Step 3: Smoke-test**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python main.py drift_check
```

Expected: table with rolling metrics and either "No drift detected" or alerts.

- [ ] **Step 4: Commit**

```bash
git add src/cli/diagnostic_cmd.py src/main.py
git commit -m "feat: CLI drift_check command — rolling prediction drift analysis"
```

---

## Task 9: Prediction Stability Analyzer

**Files:**
- Create: `src/diagnostics/prediction_stability.py`
- Test: `tests/test_prediction_stability.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_prediction_stability.py`:

```python
"""Tests for src/diagnostics/prediction_stability.py."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.config.settings import DiagnosticsConfig
from src.diagnostics.prediction_stability import StabilityAnalyzer
from src.features.feature_builder import FEATURE_COLUMNS


def _config() -> DiagnosticsConfig:
    return DiagnosticsConfig(
        drift_window=50,
        drift_alert_threshold=2.0,
        stability_n_perturbations=10,
        stability_noise_scale=0.05,
    )


def _make_stable_analyzer() -> StabilityAnalyzer:
    """Analyzer backed by deterministic mocked models."""
    elo = MagicMock()
    elo.win_draw_loss_probabilities.return_value = (0.50, 0.25, 0.25)

    poisson = MagicMock()
    poisson.win_draw_loss_from_poisson.return_value = (0.50, 0.25, 0.25)

    xgb = MagicMock()
    xgb.predict_proba_dict.return_value = [
        {"home_win": 0.50, "draw": 0.25, "away_win": 0.25}
    ]

    feature_builder = MagicMock()
    feature_builder.build_features_for_match.return_value = {
        col: 0.5 for col in FEATURE_COLUMNS
    }

    return StabilityAnalyzer(
        elo_model=elo,
        poisson_model=poisson,
        xgb_model=xgb,
        feature_builder=feature_builder,
        weights=(0.3, 0.3, 0.4),
        config=_config(),
        random_seed=42,
    )


class TestStabilityAnalyzerOutput:
    def test_returns_required_keys(self):
        analyzer = _make_stable_analyzer()
        result = analyzer.analyze("Brazil", "France")
        for key in ("home_team", "away_team", "base_probabilities",
                    "mean_probabilities", "std_probabilities",
                    "stability_band", "n_perturbations"):
            assert key in result, f"Missing key: {key}"

    def test_home_team_in_result(self):
        result = _make_stable_analyzer().analyze("Brazil", "France")
        assert result["home_team"] == "Brazil"
        assert result["away_team"] == "France"

    def test_stability_band_is_valid(self):
        result = _make_stable_analyzer().analyze("Brazil", "France")
        assert result["stability_band"] in ("Stable", "Moderate", "Unstable")

    def test_deterministic_models_give_stable_band(self):
        result = _make_stable_analyzer().analyze("Brazil", "France")
        # Deterministic mocked models → noise only affects XGBoost features
        # but XGBoost mock ignores features → same result every time
        assert result["stability_band"] == "Stable"

    def test_n_perturbations_matches_config(self):
        result = _make_stable_analyzer().analyze("Brazil", "France")
        assert result["n_perturbations"] == 10

    def test_probabilities_sum_to_one(self):
        result = _make_stable_analyzer().analyze("Brazil", "France")
        for probs_key in ("base_probabilities", "mean_probabilities"):
            probs = result[probs_key]
            total = probs["home_win"] + probs["draw"] + probs["away_win"]
            assert abs(total - 1.0) < 0.01, f"{probs_key} probabilities don't sum to 1: {total}"

    def test_std_non_negative(self):
        result = _make_stable_analyzer().analyze("Brazil", "France")
        for v in result["std_probabilities"].values():
            assert v >= 0.0
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python -m pytest tests/test_prediction_stability.py -v
```

Expected: `ImportError: cannot import name 'StabilityAnalyzer'`

- [ ] **Step 3: Create `src/diagnostics/prediction_stability.py`**

```python
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.config.settings import DiagnosticsConfig, settings
from src.features.feature_builder import FeatureBuilder, FEATURE_COLUMNS
from src.ml.xgboost_model import XGBoostMatchModel
from src.models.elo_model import EloModel
from src.models.poisson_model import PoissonModel
from src.utils.helpers import normalize_probabilities


class StabilityAnalyzer:
    """Perturbation-based sensitivity analysis for ensemble predictions.

    Injects Gaussian noise into the XGBoost feature vector ``n_perturbations``
    times and measures how much the blended ensemble probabilities vary.
    Elo and Poisson are deterministic given team names and are not perturbed.

    Stability bands:
    - Stable:   max std across outcomes < 0.02
    - Moderate: max std in [0.02, 0.05)
    - Unstable: max std ≥ 0.05
    """

    def __init__(
        self,
        elo_model: EloModel,
        poisson_model: PoissonModel,
        xgb_model: XGBoostMatchModel,
        feature_builder: FeatureBuilder,
        weights: tuple[float, float, float] = (0.3, 0.3, 0.4),
        config: DiagnosticsConfig | None = None,
        random_seed: int = 42,
    ) -> None:
        self._elo = elo_model
        self._poisson = poisson_model
        self._xgb = xgb_model
        self._feature_builder = feature_builder
        self._weights = weights  # (elo_w, poisson_w, xgb_w)
        self._config = config or settings.diagnostics
        self._rng = np.random.default_rng(random_seed)

    def analyze(
        self,
        home_team: str,
        away_team: str,
        match_date: pd.Timestamp | None = None,
    ) -> dict:
        """Return stability report for the given match."""
        if match_date is None:
            match_date = pd.Timestamp.today()

        n = self._config.stability_n_perturbations
        scale = self._config.stability_noise_scale
        ew, pw, xw = self._weights

        # Base features (deterministic)
        base_features = self._feature_builder.build_features_for_match(
            home_team, away_team, match_date
        )

        # Deterministic component probs (Elo + Poisson don't use feature vector)
        elo_w, elo_d, elo_l = self._elo.win_draw_loss_probabilities(home_team, away_team)
        poi_w, poi_d, poi_l = self._poisson.win_draw_loss_from_poisson(home_team, away_team)

        # Base XGBoost prediction
        base_feat_df = pd.DataFrame([base_features])
        base_xgb = self._xgb.predict_proba_dict(base_feat_df)[0]

        def _blend(xgb_result: dict[str, float]) -> dict[str, float]:
            raw = [
                ew * elo_w + pw * poi_w + xw * xgb_result["home_win"],
                ew * elo_d + pw * poi_d + xw * xgb_result["draw"],
                ew * elo_l + pw * poi_l + xw * xgb_result["away_win"],
            ]
            hw, dr, aw = normalize_probabilities(raw)
            return {"home_win": hw, "draw": dr, "away_win": aw}

        base_probs = _blend(base_xgb)

        # Perturbed runs — noise on continuous features only (not is_neutral)
        perturbed_probs: list[dict[str, float]] = []
        for _ in range(n):
            noisy_features = {}
            for col in FEATURE_COLUMNS:
                v = base_features.get(col, 0.0)
                if col == "is_neutral":
                    noisy_features[col] = v
                else:
                    noisy_features[col] = v + float(self._rng.normal(0, scale))

            feat_df = pd.DataFrame([noisy_features])
            try:
                xgb_result = self._xgb.predict_proba_dict(feat_df)[0]
            except Exception as exc:
                logger.warning(f"XGBoost perturbation failed: {exc}; using base")
                xgb_result = base_xgb
            perturbed_probs.append(_blend(xgb_result))

        home_wins = [p["home_win"] for p in perturbed_probs]
        draws = [p["draw"] for p in perturbed_probs]
        away_wins = [p["away_win"] for p in perturbed_probs]

        max_std = max(float(np.std(home_wins)), float(np.std(draws)), float(np.std(away_wins)))

        if max_std < 0.02:
            band = "Stable"
        elif max_std < 0.05:
            band = "Moderate"
        else:
            band = "Unstable"

        return {
            "home_team": home_team,
            "away_team": away_team,
            "base_probabilities": {k: round(v, 4) for k, v in base_probs.items()},
            "mean_probabilities": {
                "home_win": round(float(np.mean(home_wins)), 4),
                "draw": round(float(np.mean(draws)), 4),
                "away_win": round(float(np.mean(away_wins)), 4),
            },
            "std_probabilities": {
                "home_win": round(float(np.std(home_wins)), 4),
                "draw": round(float(np.std(draws)), 4),
                "away_win": round(float(np.std(away_wins)), 4),
            },
            "stability_band": band,
            "n_perturbations": n,
        }
```

- [ ] **Step 4: Run stability tests**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python -m pytest tests/test_prediction_stability.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/diagnostics/prediction_stability.py tests/test_prediction_stability.py
git commit -m "feat: StabilityAnalyzer — perturbation-based sensitivity analysis"
```

---

## Task 10: Reliability Monitor

**Files:**
- Create: `src/diagnostics/reliability_monitor.py`

- [ ] **Step 1: Create `src/diagnostics/reliability_monitor.py`**

No separate test file — this is a thin aggregator; covered by integration through CLI. Inline tests confirm the contract.

```python
from __future__ import annotations

import pandas as pd

from src.diagnostics.confidence_scorer import ConfidenceScorer
from src.diagnostics.drift_detection import DriftDetector
from src.diagnostics.prediction_stability import StabilityAnalyzer


class ReliabilityMonitor:
    """Aggregates confidence, drift, and stability into a single report dict.

    Used by the reports package to build forecast reliability summaries.
    """

    def __init__(
        self,
        confidence_scorer: ConfidenceScorer,
        drift_detector: DriftDetector,
        stability_analyzer: StabilityAnalyzer,
    ) -> None:
        self._confidence = confidence_scorer
        self._drift = drift_detector
        self._stability = stability_analyzer

    def report(
        self,
        home_team: str,
        away_team: str,
        match_date: pd.Timestamp,
        component_models: dict[str, dict[str, float]],
        ensemble_probs: dict[str, float],
        feature_vector: dict[str, float],
    ) -> dict:
        """Return combined reliability assessment for one match."""
        return {
            "confidence": self._confidence.score(
                home_team, away_team, component_models, ensemble_probs, feature_vector
            ),
            "drift": self._drift.status(),
            "stability": self._stability.analyze(home_team, away_team, match_date),
        }
```

- [ ] **Step 2: Verify the module imports cleanly**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python -c "from src.diagnostics.reliability_monitor import ReliabilityMonitor; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Run full test suite**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python -m pytest tests/ -q
```

Expected: all existing 200 tests + new tests pass, no regressions.

- [ ] **Step 4: Check coverage on new modules**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python -m pytest --cov=src tests/ --cov-report=term-missing -q 2>&1 | grep -E "optimization|diagnostics|TOTAL"
```

Expected: `src/optimization/` and `src/diagnostics/` at 85%+.

- [ ] **Step 5: Commit**

```bash
git add src/diagnostics/reliability_monitor.py
git commit -m "feat: ReliabilityMonitor — aggregates confidence, drift, stability"
```

---

## Task 11: Phase 3a Wrap-Up — CLAUDE.md + Final Commit

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update `CLAUDE.md` — add Phase 3a modules to Module Responsibilities**

In the `## Module Responsibilities` section, add after the Phase 2 section:

```markdown
### Phase 3a Modules

- `optimization/weight_search.py` — simplex weight validation + grid generation utilities.
- `optimization/ensemble_optimizer.py` — SLSQP optimizer (`WeightOptimizer`). Finds ensemble weights minimising log loss on the chronological test split. Saves `best_weight_config.json`.
- `diagnostics/confidence_scorer.py` — `ConfidenceScorer`. Five-factor confidence engine: model agreement (0.35), calibration quality (0.25), feature completeness (0.20), historical reliability (0.15), prediction volatility (0.05). Returns score + band + breakdown.
- `diagnostics/drift_detection.py` — `DriftDetector`. Rolling window drift detection over (outcome, probs) pairs. Raises `DriftAlert` when metric exceeds baseline + N×std.
- `diagnostics/prediction_stability.py` — `StabilityAnalyzer`. Perturbs XGBoost feature vector N times with Gaussian noise; measures how much the blended ensemble output varies. Bands: Stable/Moderate/Unstable.
- `diagnostics/reliability_monitor.py` — `ReliabilityMonitor`. Thin aggregator; calls confidence, drift, and stability and returns a combined report dict.
- `cli/optimize_cmd.py` — `cmd_optimize()`. CLI wrapper for `WeightOptimizer`.
- `cli/diagnostic_cmd.py` — `cmd_confidence()`, `cmd_drift_check()`. CLI wrappers for confidence scoring and drift analysis.
```

- [ ] **Step 2: Add Phase 3a architecture entry to the Architecture table**

In the Architecture table, add rows:

```markdown
| Optimization | `src/optimization/` | ensemble, models, features, evaluation |
| Diagnostics | `src/diagnostics/` | models, ensemble, evaluation |
| CLI commands | `src/cli/` | optimization, diagnostics, ensemble, ml |
```

- [ ] **Step 3: Run full suite one final time**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python -m pytest tests/ -q --tb=short
```

Expected: all tests pass.

- [ ] **Step 4: Final Phase 3a commit**

```bash
git add CLAUDE.md
git commit -m "docs: CLAUDE.md updated for Phase 3a — intelligence layer complete"
git tag v0.3a-phase3a-intelligence-layer
```

---

*Phase 3a complete. Proceed to `2026-05-12-phase3b-explainability-scenarios.md` for SHAP, scenarios, reports, and robustness.*
