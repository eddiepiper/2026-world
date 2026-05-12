# Phase 3b — Explainability, Scenarios & Reports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SHAP-based explainability (optional dep), scenario analysis, upset flagging, robustness stress tests, and auto-generated markdown forecast reports to the 2026-world forecasting system.

**Architecture:** Two new packages (`src/explainability/`, `src/scenarios/`) plus `src/reports/` feed into three new CLI commands. `src/explainability/` is fully standalone — it imports only from `src/ml/` and `src/features/` and degrades gracefully if `shap` is not installed. `src/scenarios/` imports from `src/diagnostics/` for confidence scoring.

**Prerequisite:** Phase 3a complete (DiagnosticsConfig in settings, diagnostics package exists, cli package exists).

**Tech Stack:** shap (optional — install separately with `pip install shap`), scipy (already installed), numpy, pandas, loguru, rich.

---

## File Map

**Create:**
- `src/explainability/__init__.py`
- `src/explainability/shap_engine.py` — TreeExplainer wrapper with optional-dep guard
- `src/explainability/feature_impact.py` — global importance ranking from SHAP values
- `src/explainability/prediction_explainer.py` — per-match "top drivers" formatter
- `src/scenarios/__init__.py`
- `src/scenarios/scenario_runner.py` — apply param perturbations, re-run ensemble
- `src/scenarios/upset_analysis.py` — flag high-uncertainty / overvalued-favourite matches
- `src/scenarios/stress_tests.py` — missing features, noisy inputs, edge-case robustness
- `src/reports/__init__.py`
- `src/reports/markdown_reporter.py` — render diagnostics + SHAP + upsets to .md
- `src/reports/forecast_summary.py` — top-level `generate_report()` entry point
- `src/cli/explain_cmd.py`
- `src/cli/scenario_cmd.py`
- `src/cli/report_cmd.py`
- `tests/test_shap_engine.py`
- `tests/test_scenario_runner.py`
- `tests/test_forecast_summary.py`

**Modify:**
- `src/main.py` — add `explain`, `scenarios`, `report` commands

---

## Task 1: Explainability Package — shap_engine.py

**Files:**
- Create: `src/explainability/__init__.py`
- Create: `src/explainability/shap_engine.py`
- Test: `tests/test_shap_engine.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_shap_engine.py`:

```python
"""Tests for src/explainability/shap_engine.py.

shap.TreeExplainer is always mocked — tests must pass whether or not
shap is installed, and must explicitly test graceful degradation.
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
    inner = MagicMock()
    inner.get_booster.return_value = MagicMock()
    model.pipeline = MagicMock()
    model.pipeline.__getitem__ = MagicMock(return_value=inner)
    return model


def _shap_values_fixture(n: int = 5) -> np.ndarray:
    """Return shape (n, n_features, 3) SHAP values array."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, len(FEATURE_COLUMNS), 3))


class TestSHAPUnavailable:
    """When _SHAP_AVAILABLE is False, public methods raise ImportError."""

    def test_local_shap_raises_import_error_when_unavailable(self):
        with patch("src.explainability.shap_engine._SHAP_AVAILABLE", False):
            from src.explainability.shap_engine import SHAPEngine
            engine = SHAPEngine.__new__(SHAPEngine)
            with pytest.raises(ImportError, match="shap"):
                engine.local_shap(pd.DataFrame([{col: 0.5 for col in FEATURE_COLUMNS}]))

    def test_global_shap_raises_import_error_when_unavailable(self):
        with patch("src.explainability.shap_engine._SHAP_AVAILABLE", False):
            from src.explainability.shap_engine import SHAPEngine
            engine = SHAPEngine.__new__(SHAPEngine)
            with pytest.raises(ImportError, match="shap"):
                engine.global_shap(pd.DataFrame([{col: 0.5 for col in FEATURE_COLUMNS}]))


class TestSHAPEngineWithMockedSHAP:
    """Test SHAPEngine behaviour when shap IS available (mocked)."""

    def _make_engine(self):
        xgb = _make_xgb_model()
        mock_shap = MagicMock()
        explainer = MagicMock()
        explainer.shap_values.return_value = _shap_values_fixture(n=5)
        mock_shap.TreeExplainer.return_value = explainer

        with patch.dict("sys.modules", {"shap": mock_shap}):
            with patch("src.explainability.shap_engine._SHAP_AVAILABLE", True):
                from importlib import reload
                import src.explainability.shap_engine as mod
                # Inject mock shap into module namespace
                mod.shap = mock_shap
                engine = mod.SHAPEngine(xgb_model=xgb)
                engine._explainer = explainer
                return engine

    def test_global_shap_returns_dict_with_feature_names(self):
        engine = self._make_engine()
        X = pd.DataFrame([{col: 0.5 for col in FEATURE_COLUMNS}] * 5)
        result = engine.global_shap(X)
        assert isinstance(result, dict)
        assert all(col in result for col in FEATURE_COLUMNS)

    def test_global_shap_values_non_negative(self):
        engine = self._make_engine()
        X = pd.DataFrame([{col: 0.5 for col in FEATURE_COLUMNS}] * 5)
        result = engine.global_shap(X)
        assert all(v >= 0 for v in result.values())

    def test_local_shap_returns_list(self):
        engine = self._make_engine()
        X = pd.DataFrame([{col: 0.5 for col in FEATURE_COLUMNS}])
        engine._explainer.shap_values.return_value = _shap_values_fixture(n=1)
        result = engine.local_shap(X)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_local_shap_entries_have_feature_and_value(self):
        engine = self._make_engine()
        X = pd.DataFrame([{col: 0.5 for col in FEATURE_COLUMNS}])
        engine._explainer.shap_values.return_value = _shap_values_fixture(n=1)
        result = engine.local_shap(X)
        for entry in result:
            assert "feature" in entry
            assert "shap_value" in entry
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python -m pytest tests/test_shap_engine.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.explainability'`

- [ ] **Step 3: Create `src/explainability/__init__.py`** (empty)

```python
```

- [ ] **Step 4: Create `src/explainability/shap_engine.py`**

```python
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

from src.features.feature_builder import FEATURE_COLUMNS
from src.ml.xgboost_model import XGBoostMatchModel

_SHAP_UNAVAILABLE_MSG = (
    "Explainability unavailable: shap is not installed. "
    "Install it with: pip install shap"
)


class SHAPEngine:
    """SHAP-based explainability for the XGBoost component model.

    Requires shap to be installed. If shap is unavailable, all public
    methods raise ImportError with a user-friendly install hint.

    Usage::

        engine = SHAPEngine(xgb_model)
        global_imp = engine.global_shap(X_test)   # {feature: mean_abs_shap}
        local_imp  = engine.local_shap(X_single)  # [{feature, shap_value}, ...]
    """

    def __init__(self, xgb_model: XGBoostMatchModel) -> None:
        self._model = xgb_model
        self._explainer = None  # lazy-initialised on first call

    def _get_explainer(self):
        if not _SHAP_AVAILABLE:
            raise ImportError(_SHAP_UNAVAILABLE_MSG)
        if self._explainer is None:
            booster = self._model.pipeline["xgb"].get_booster()
            self._explainer = shap.TreeExplainer(booster)
            logger.info("SHAP TreeExplainer initialised.")
        return self._explainer

    def global_shap(self, X: pd.DataFrame) -> dict[str, float]:
        """Compute mean |SHAP value| per feature across all outcome classes.

        Returns dict mapping feature name → mean absolute SHAP value.
        Higher means more globally important.

        Parameters
        ----------
        X:
            Feature matrix matching FEATURE_COLUMNS. Use the test split
            (post test_split_date) for a meaningful global importance.
        """
        explainer = self._get_explainer()
        shap_values = explainer.shap_values(X)  # shape (n, n_features) or (n, n_features, n_classes)

        # TreeExplainer for multiclass returns (n_samples, n_features, n_classes)
        if isinstance(shap_values, list):
            # older shap versions return list of per-class arrays
            arr = np.stack(shap_values, axis=-1)  # (n, n_features, n_classes)
        else:
            arr = shap_values  # (n, n_features, n_classes) or (n, n_features)

        if arr.ndim == 3:
            mean_abs = np.mean(np.abs(arr), axis=(0, 2))  # (n_features,)
        else:
            mean_abs = np.mean(np.abs(arr), axis=0)  # (n_features,)

        feature_names = self._model.feature_names or FEATURE_COLUMNS
        return {
            name: float(round(val, 6))
            for name, val in zip(feature_names, mean_abs)
        }

    def local_shap(
        self,
        X: pd.DataFrame,
        outcome: str = "H",
    ) -> list[dict]:
        """Compute per-feature SHAP values for a single match (local explanation).

        Returns list of dicts sorted by |shap_value| descending::

            [{"feature": "elo_diff", "shap_value": 0.18, "raw_value": 120.5}, ...]

        Parameters
        ----------
        X:
            Single-row feature DataFrame.
        outcome:
            Which class to explain: "H" (home win), "D" (draw), "A" (away win).
        """
        explainer = self._get_explainer()
        shap_values = explainer.shap_values(X)

        classes = ["A", "D", "H"]
        class_idx = classes.index(outcome) if outcome in classes else 2

        if isinstance(shap_values, list):
            arr = shap_values[class_idx]  # (1, n_features)
        else:
            arr = shap_values[:, :, class_idx] if shap_values.ndim == 3 else shap_values

        feature_names = self._model.feature_names or FEATURE_COLUMNS
        row = arr[0] if arr.ndim == 2 else arr
        raw_values = X.iloc[0].to_dict()

        result = [
            {
                "feature": name,
                "shap_value": float(round(val, 6)),
                "raw_value": float(raw_values.get(name, 0.0)),
            }
            for name, val in zip(feature_names, row)
        ]
        result.sort(key=lambda d: abs(d["shap_value"]), reverse=True)
        return result
```

- [ ] **Step 5: Run SHAP engine tests**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python -m pytest tests/test_shap_engine.py -v
```

Expected: all tests pass (mocked SHAP throughout — no real shap needed).

- [ ] **Step 6: Commit**

```bash
git add src/explainability/__init__.py src/explainability/shap_engine.py tests/test_shap_engine.py
git commit -m "feat: explainability package — SHAPEngine with optional-dep guard"
```

---

## Task 2: feature_impact.py + prediction_explainer.py

**Files:**
- Create: `src/explainability/feature_impact.py`
- Create: `src/explainability/prediction_explainer.py`

- [ ] **Step 1: Create `src/explainability/feature_impact.py`**

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from src.explainability.shap_engine import SHAPEngine


class FeatureImpactRanker:
    """Rank features by global SHAP importance and optionally save to file."""

    def __init__(self, shap_engine: SHAPEngine) -> None:
        self._engine = shap_engine

    def rank(self, X: pd.DataFrame, top_n: int = 10) -> list[dict]:
        """Return top_n features sorted by mean |SHAP|.

        Returns list of dicts: [{"rank": 1, "feature": "elo_diff", "importance": 0.12}, ...]
        """
        global_imp = self._engine.global_shap(X)
        sorted_features = sorted(global_imp.items(), key=lambda x: x[1], reverse=True)
        return [
            {"rank": i + 1, "feature": name, "importance": round(val, 6)}
            for i, (name, val) in enumerate(sorted_features[:top_n])
        ]

    def to_markdown(self, ranked: list[dict]) -> str:
        lines = ["## Global Feature Importance (SHAP)\n"]
        lines.append("| Rank | Feature | Mean |SHAP| |")
        lines.append("|------|---------|--------------|")
        for entry in ranked:
            lines.append(f"| {entry['rank']} | {entry['feature']} | {entry['importance']:.6f} |")
        return "\n".join(lines)
```

- [ ] **Step 2: Create `src/explainability/prediction_explainer.py`**

```python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from src.explainability.shap_engine import SHAPEngine


class PredictionExplainer:
    """Format local SHAP values into human-readable explanations.

    Saves both markdown and JSON to outputs/reports/explanations/.
    """

    def __init__(self, shap_engine: SHAPEngine) -> None:
        self._engine = shap_engine

    def explain(
        self,
        home_team: str,
        away_team: str,
        feature_row: pd.DataFrame,
        outcome: str = "H",
        top_n: int = 5,
    ) -> dict:
        """Return top-N SHAP drivers for the given match and outcome.

        Returns::

            {
                "home_team": "Brazil",
                "away_team": "France",
                "outcome_explained": "H",
                "top_drivers": [
                    {"rank": 1, "feature": "elo_diff", "shap_value": 0.18, "raw_value": 120.5},
                    ...
                ],
            }
        """
        local = self._engine.local_shap(feature_row, outcome=outcome)
        return {
            "home_team": home_team,
            "away_team": away_team,
            "outcome_explained": outcome,
            "top_drivers": [
                {"rank": i + 1, **entry}
                for i, entry in enumerate(local[:top_n])
            ],
        }

    def to_markdown(self, explanation: dict) -> str:
        home = explanation["home_team"]
        away = explanation["away_team"]
        outcome_label = {"H": "Home win", "D": "Draw", "A": "Away win"}.get(
            explanation["outcome_explained"], explanation["outcome_explained"]
        )
        lines = [
            f"## {home} vs {away} — Top Drivers ({outcome_label})\n",
        ]
        for driver in explanation["top_drivers"]:
            sign = "+" if driver["shap_value"] >= 0 else ""
            lines.append(
                f"{driver['rank']}. {driver['feature']:<28} "
                f"{sign}{driver['shap_value']:.4f}"
            )
        return "\n".join(lines)

    def save(
        self,
        explanation: dict,
        output_dir: Path,
        match_date: str | None = None,
    ) -> None:
        """Save explanation as both .md and .json."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        home = explanation["home_team"].replace(" ", "_")
        away = explanation["away_team"].replace(" ", "_")
        date_str = match_date or pd.Timestamp.today().strftime("%Y-%m-%d")
        stem = f"{home}_vs_{away}_{date_str}"

        md_path = output_dir / f"{stem}.md"
        json_path = output_dir / f"{stem}.json"

        md_path.write_text(self.to_markdown(explanation))
        json_path.write_text(json.dumps(explanation, indent=2))

        logger.info(f"Explanation saved to {md_path} and {json_path}")
```

- [ ] **Step 3: Verify modules import cleanly**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python -c "
from src.explainability.feature_impact import FeatureImpactRanker
from src.explainability.prediction_explainer import PredictionExplainer
print('OK')
"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/explainability/feature_impact.py src/explainability/prediction_explainer.py
git commit -m "feat: feature_impact + prediction_explainer — SHAP formatting layer"
```

---

## Task 3: CLI `explain` Command

**Files:**
- Create: `src/cli/explain_cmd.py`
- Modify: `src/main.py`

- [ ] **Step 1: Create `src/cli/explain_cmd.py`**

```python
from __future__ import annotations

import sys

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from src.config.settings import settings
from src.ingestion.match_loader import load_matches
from src.models.elo_model import EloModel
from src.models.poisson_model import PoissonModel

console = Console()


def cmd_explain(home_team: str, away_team: str, outcome: str = "H") -> None:
    """Print SHAP-based top drivers for the given match. Requires shap + trained model."""
    from src.explainability.shap_engine import SHAPEngine, _SHAP_UNAVAILABLE_MSG, _SHAP_AVAILABLE
    from src.explainability.prediction_explainer import PredictionExplainer
    from src.features.feature_builder import FeatureBuilder
    from src.ml.xgboost_model import XGBoostMatchModel

    if not _SHAP_AVAILABLE:
        console.print(f"[bold yellow]{_SHAP_UNAVAILABLE_MSG}[/bold yellow]")
        sys.exit(0)

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
        matches_df=matches_df, elo_model=elo, poisson_model=poisson
    )

    match_date = pd.Timestamp.today()
    features = feature_builder.build_features_for_match(home_team, away_team, match_date)
    feature_row = pd.DataFrame([features])

    console.print(Panel(
        f"Computing SHAP explanation for {home_team} vs {away_team}…",
        style="blue"
    ))

    try:
        shap_engine = SHAPEngine(xgb_model=xgb_model)
        explainer = PredictionExplainer(shap_engine=shap_engine)
        explanation = explainer.explain(
            home_team=home_team,
            away_team=away_team,
            feature_row=feature_row,
            outcome=outcome,
            top_n=5,
        )

        outcome_label = {"H": "Home win", "D": "Draw", "A": "Away win"}.get(outcome, outcome)
        console.print(f"\n[bold]{home_team} vs {away_team}[/bold] — Top drivers ({outcome_label}):\n")

        for driver in explanation["top_drivers"]:
            sign = "+" if driver["shap_value"] >= 0 else ""
            color = "green" if driver["shap_value"] >= 0 else "red"
            console.print(
                f"  {driver['rank']}. {driver['feature']:<28} "
                f"[{color}]{sign}{driver['shap_value']:.4f}[/{color}]"
            )

        out_dir = settings.outputs_dir / "reports" / "explanations"
        explainer.save(explanation, out_dir, match_date.strftime("%Y-%m-%d"))
        console.print(f"\n[dim]Saved to {out_dir}[/dim]\n")

    except ImportError as exc:
        console.print(f"[bold yellow]{exc}[/bold yellow]")
        sys.exit(0)
    except Exception as exc:
        logger.error(f"SHAP explanation failed: {exc}")
        console.print(f"[red]Explanation failed: {exc}[/red]")
        sys.exit(1)
```

- [ ] **Step 2: Wire `explain` into `src/main.py`**

Add import:

```python
from src.cli.explain_cmd import cmd_explain
```

Add branch in `main()`:

```python
    elif command == "explain":
        if len(args) < 3:
            console.print("[red]explain requires at least two team names[/red]")
            sys.exit(1)
        outcome = args[3] if len(args) > 3 else "H"
        cmd_explain(args[1], args[2], outcome)
```

Add usage line:

```python
        console.print("  python main.py explain <home_team> <away_team> [H|D|A]")
```

- [ ] **Step 3: Smoke-test graceful degradation (shap not installed)**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python main.py explain Brazil France
```

Expected: prints `"Explainability unavailable: shap is not installed…"` and exits cleanly with code 0 (not a crash).

- [ ] **Step 4: Commit**

```bash
git add src/cli/explain_cmd.py src/main.py
git commit -m "feat: CLI explain command — SHAP top drivers with graceful degradation"
```

---

## Task 4: Scenario Runner

**Files:**
- Create: `src/scenarios/__init__.py`
- Create: `src/scenarios/scenario_runner.py`
- Test: `tests/test_scenario_runner.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_scenario_runner.py`:

```python
"""Tests for src/scenarios/scenario_runner.py and upset_analysis.py."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.scenarios.scenario_runner import ScenarioRunner, Perturbation


def _make_elo(rating: float = 1600.0) -> MagicMock:
    elo = MagicMock()
    elo.ratings = {"Brazil": rating, "France": 1550.0}  # real dict — ScenarioRunner restores via ratings[team]
    elo.get_team_rating.return_value = rating
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
    from src.ensemble.ensemble_engine import EnsembleWeights
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
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python -m pytest tests/test_scenario_runner.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.scenarios'`

- [ ] **Step 3: Create `src/scenarios/__init__.py`** (empty)

```python
```

- [ ] **Step 4: Create `src/scenarios/scenario_runner.py`**

```python
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from loguru import logger

from src.features.feature_builder import FeatureBuilder, FEATURE_COLUMNS
from src.ml.xgboost_model import XGBoostMatchModel
from src.models.elo_model import EloModel
from src.models.poisson_model import PoissonModel
from src.utils.helpers import normalize_probabilities

_VALID_PARAMS = frozenset(
    {"attack_strength", "defense_strength", "elo_rating", "recent_form"}
)


@dataclass
class Perturbation:
    """A single named adjustment to a team's strength parameter.

    Example::

        Perturbation(param="attack_strength", team="France", delta=-0.3)
    """

    param: str
    team: str
    delta: float

    def validate(self) -> None:
        if self.param not in _VALID_PARAMS:
            raise ValueError(
                f"Unknown param '{self.param}'. "
                f"Valid options: {sorted(_VALID_PARAMS)}"
            )


class ScenarioRunner:
    """Apply parameter perturbations and measure probability delta.

    Supported params:
    - ``attack_strength``  — scale team's Poisson attack strength
    - ``defense_strength`` — scale team's Poisson defense strength
    - ``elo_rating``       — shift team's Elo rating
    - ``recent_form``      — shift home_win_rate or away_win_rate feature
    """

    def __init__(
        self,
        elo_model: EloModel,
        poisson_model: PoissonModel,
        xgb_model: XGBoostMatchModel,
        feature_builder: FeatureBuilder,
        weights: tuple[float, float, float] = (0.3, 0.3, 0.4),
    ) -> None:
        self._elo = elo_model
        self._poisson = poisson_model
        self._xgb = xgb_model
        self._feature_builder = feature_builder
        self._weights = weights

    def run(
        self,
        home_team: str,
        away_team: str,
        perturbations: list[Perturbation],
        match_date: pd.Timestamp | None = None,
    ) -> dict:
        """Return a delta table: original vs perturbed ensemble probabilities."""
        for p in perturbations:
            p.validate()

        if match_date is None:
            match_date = pd.Timestamp.today()

        baseline = self._blend(home_team, away_team, match_date)

        # Apply perturbations (shallow copies — restore after)
        saved_elo = {t: self._elo.get_team_rating(t)
                     for p in perturbations for t in [p.team]}
        saved_attack = dict(self._poisson.attack_strength)
        saved_defense = dict(self._poisson.defense_strength)

        for p in perturbations:
            if p.param == "elo_rating":
                old = self._elo.get_team_rating(p.team)
                self._elo.ratings[p.team] = old + p.delta
            elif p.param == "attack_strength":
                self._poisson.attack_strength[p.team] = (
                    self._poisson.attack_strength.get(p.team, 1.0) + p.delta
                )
            elif p.param == "defense_strength":
                self._poisson.defense_strength[p.team] = (
                    self._poisson.defense_strength.get(p.team, 1.0) + p.delta
                )
            # recent_form handled via feature override below

        form_deltas = {
            p.team: p.delta for p in perturbations if p.param == "recent_form"
        }
        perturbed = self._blend(
            home_team, away_team, match_date, form_deltas=form_deltas
        )

        # Restore
        for team, rating in saved_elo.items():
            self._elo.ratings[team] = rating
        self._poisson.attack_strength = saved_attack
        self._poisson.defense_strength = saved_defense

        delta = {
            k: round(perturbed[k] - baseline[k], 4)
            for k in ("home_win", "draw", "away_win")
        }

        return {
            "home_team": home_team,
            "away_team": away_team,
            "baseline": {k: round(v, 4) for k, v in baseline.items()},
            "perturbed": {k: round(v, 4) for k, v in perturbed.items()},
            "delta": delta,
            "perturbations": [
                {"param": p.param, "team": p.team, "delta": p.delta}
                for p in perturbations
            ],
        }

    def to_markdown(self, result: dict) -> str:
        home, away = result["home_team"], result["away_team"]
        lines = [
            f"## Scenario: {home} vs {away}\n",
            "| Outcome | Baseline | Perturbed | Delta |",
            "|---------|----------|-----------|-------|",
        ]
        for outcome in ("home_win", "draw", "away_win"):
            label = outcome.replace("_", " ").title()
            b = result["baseline"][outcome]
            p = result["perturbed"][outcome]
            d = result["delta"][outcome]
            sign = "+" if d >= 0 else ""
            lines.append(f"| {label} | {b:.3f} | {p:.3f} | {sign}{d:.3f} |")

        if result["perturbations"]:
            lines.append("\n**Perturbations applied:**")
            for pt in result["perturbations"]:
                sign = "+" if pt["delta"] >= 0 else ""
                lines.append(f"- {pt['team']} {pt['param']}: {sign}{pt['delta']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _blend(
        self,
        home_team: str,
        away_team: str,
        match_date: pd.Timestamp,
        form_deltas: dict[str, float] | None = None,
    ) -> dict[str, float]:
        ew, pw, xw = self._weights

        elo_w, elo_d, elo_l = self._elo.win_draw_loss_probabilities(home_team, away_team)
        poi_w, poi_d, poi_l = self._poisson.win_draw_loss_from_poisson(home_team, away_team)

        features = self._feature_builder.build_features_for_match(
            home_team, away_team, match_date
        )

        # Apply recent_form perturbation to feature dict
        if form_deltas:
            if home_team in form_deltas:
                features["home_win_rate"] = (
                    features.get("home_win_rate", 0.0) + form_deltas[home_team]
                )
            if away_team in form_deltas:
                features["away_win_rate"] = (
                    features.get("away_win_rate", 0.0) + form_deltas[away_team]
                )

        feat_df = pd.DataFrame([features])
        xgb_result = self._xgb.predict_proba_dict(feat_df)[0]

        raw = [
            ew * elo_w + pw * poi_w + xw * xgb_result["home_win"],
            ew * elo_d + pw * poi_d + xw * xgb_result["draw"],
            ew * elo_l + pw * poi_l + xw * xgb_result["away_win"],
        ]
        hw, dr, aw = normalize_probabilities(raw)
        return {"home_win": hw, "draw": dr, "away_win": aw}
```

- [ ] **Step 5: Run scenario runner tests**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python -m pytest tests/test_scenario_runner.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/scenarios/__init__.py src/scenarios/scenario_runner.py tests/test_scenario_runner.py
git commit -m "feat: ScenarioRunner — perturbation-based probability delta analysis"
```

---

## Task 5: Upset Analysis

**Files:**
- Create: `src/scenarios/upset_analysis.py`
- Test: `tests/test_scenario_runner.py` (extend)

- [ ] **Step 1: Write failing tests for UpsetAnalyzer**

Append to `tests/test_scenario_runner.py`:

```python
from src.scenarios.upset_analysis import UpsetAnalyzer, UpsetFlag


class TestUpsetAnalyzer:
    def test_clear_favourite_not_flagged(self):
        analyzer = UpsetAnalyzer()
        flag = analyzer.check(
            home_team="Brazil",
            away_team="Andorra",
            home_win_prob=0.85,
            away_win_prob=0.05,
            confidence_band="High",
            stability_band="Stable",
        )
        assert not flag.is_upset

    def test_underdog_above_threshold_is_flagged(self):
        analyzer = UpsetAnalyzer(underdog_threshold=0.35)
        flag = analyzer.check(
            home_team="Brazil",
            away_team="France",
            home_win_prob=0.40,
            away_win_prob=0.38,
            confidence_band="High",
            stability_band="Stable",
        )
        assert flag.is_upset

    def test_low_confidence_triggers_flag(self):
        analyzer = UpsetAnalyzer()
        flag = analyzer.check(
            home_team="Brazil",
            away_team="France",
            home_win_prob=0.60,
            away_win_prob=0.20,
            confidence_band="Low",
            stability_band="Stable",
        )
        assert flag.is_upset

    def test_unstable_prediction_triggers_flag(self):
        analyzer = UpsetAnalyzer()
        flag = analyzer.check(
            home_team="Brazil",
            away_team="France",
            home_win_prob=0.60,
            away_win_prob=0.20,
            confidence_band="High",
            stability_band="Unstable",
        )
        assert flag.is_upset

    def test_flag_has_reason(self):
        analyzer = UpsetAnalyzer(underdog_threshold=0.35)
        flag = analyzer.check(
            home_team="Brazil",
            away_team="France",
            home_win_prob=0.40,
            away_win_prob=0.38,
            confidence_band="High",
            stability_band="Stable",
        )
        assert flag.is_upset
        assert len(flag.reasons) > 0
```

- [ ] **Step 2: Run to confirm new tests fail**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python -m pytest tests/test_scenario_runner.py::TestUpsetAnalyzer -v
```

Expected: `ImportError: cannot import name 'UpsetAnalyzer'`

- [ ] **Step 3: Create `src/scenarios/upset_analysis.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class UpsetFlag:
    home_team: str
    away_team: str
    is_upset: bool
    reasons: list[str] = field(default_factory=list)


class UpsetAnalyzer:
    """Flag matches with meaningful upset potential or high forecast uncertainty.

    A match is flagged if ANY of the following apply:
    - The underdog's win probability exceeds ``underdog_threshold``
    - The confidence band is "Low"
    - The stability band is "Unstable"
    """

    def __init__(self, underdog_threshold: float = 0.35) -> None:
        self._threshold = underdog_threshold

    def check(
        self,
        home_team: str,
        away_team: str,
        home_win_prob: float,
        away_win_prob: float,
        confidence_band: str,
        stability_band: str,
    ) -> UpsetFlag:
        reasons: list[str] = []

        # Determine underdog and their win probability
        if home_win_prob < away_win_prob:
            underdog, underdog_prob = home_team, home_win_prob
        else:
            underdog, underdog_prob = away_team, away_win_prob

        if underdog_prob > self._threshold:
            reasons.append(
                f"{underdog} win probability {underdog_prob:.1%} "
                f"exceeds underdog threshold {self._threshold:.0%}"
            )
        if confidence_band == "Low":
            reasons.append("Low confidence band — prediction reliability is poor")
        if stability_band == "Unstable":
            reasons.append("Unstable prediction — high sensitivity to input variation")

        return UpsetFlag(
            home_team=home_team,
            away_team=away_team,
            is_upset=len(reasons) > 0,
            reasons=reasons,
        )
```

- [ ] **Step 4: Run all scenario tests**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python -m pytest tests/test_scenario_runner.py -v
```

Expected: all tests pass (ScenarioRunner + UpsetAnalyzer).

- [ ] **Step 5: Commit**

```bash
git add src/scenarios/upset_analysis.py tests/test_scenario_runner.py
git commit -m "feat: UpsetAnalyzer — flag high-uncertainty and underdog-potential matches"
```

---

## Task 6: Stress Tests

**Files:**
- Create: `src/scenarios/stress_tests.py`

- [ ] **Step 1: Create `src/scenarios/stress_tests.py`**

```python
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
    1. Missing features (zero-fill each column in turn)
    2. Noisy inputs (±20% uniform noise on all continuous features)
    3. Low-data teams (features all set to global fallback values)
    4. Extreme probabilities via near-zero features

    Goal: system must not raise unhandled exceptions. Probabilities must
    remain in [0, 1] and sum to ~1.0.
    """

    def __init__(self, xgb_model: XGBoostMatchModel, random_seed: int = 42) -> None:
        self._model = xgb_model
        self._rng = np.random.default_rng(random_seed)

    def run_all(self, base_features: dict[str, float]) -> list[StressTestResult]:
        """Run all stress tests and return results."""
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

    # ------------------------------------------------------------------
    # Individual tests
    # ------------------------------------------------------------------

    def _predict_safe(self, features: dict[str, float]) -> StressTestResult | None:
        """Return None if prediction succeeds with valid probs, else StressTestResult(failed)."""
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
            return None  # success
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
            return StressTestResult(
                test_name=name, passed=False,
                error=f"{len(failures)} columns caused failures"
            )
        return StressTestResult(
            test_name=name, passed=True,
            details=f"Zero-filled {len(FEATURE_COLUMNS)} columns — all predictions valid"
        )

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
                return StressTestResult(
                    test_name=name, passed=False, error=f"Trial {trial}: {err.error}"
                )
        return StressTestResult(
            test_name=name, passed=True,
            details="10 noisy-input trials — all predictions valid"
        )

    def _test_low_data_team(self, base: dict[str, float]) -> StressTestResult:
        name = "low_data_team"
        # Simulate a team with minimal historical data by zeroing out computed features
        sparse = {
            col: (base.get(col, 0.0) if col in ("home_elo", "away_elo", "elo_diff") else 0.0)
            for col in FEATURE_COLUMNS
        }
        err = self._predict_safe(sparse)
        if err:
            return StressTestResult(test_name=name, passed=False, error=err.error)
        return StressTestResult(
            test_name=name, passed=True,
            details="Sparse feature vector (low-data team) — prediction valid"
        )

    def _test_extreme_features(self, base: dict[str, float]) -> StressTestResult:
        name = "extreme_features"
        # Very high Elo difference (dominant team)
        extreme = {**base, "elo_diff": 800.0, "home_elo": 2200.0, "away_elo": 1200.0}
        err = self._predict_safe(extreme)
        if err:
            return StressTestResult(test_name=name, passed=False, error=err.error)
        return StressTestResult(
            test_name=name, passed=True,
            details="Extreme Elo gap (800 pts) — prediction valid and in [0,1]"
        )
```

- [ ] **Step 2: Write tests for RobustnessTester**

Append to `tests/test_scenario_runner.py`:

```python
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
```

- [ ] **Step 3: Run robustness tests**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python -m pytest tests/test_scenario_runner.py::TestRobustnessTester -v
```

Expected: all 5 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/scenarios/stress_tests.py tests/test_scenario_runner.py
git commit -m "feat: RobustnessTester — stress tests for missing/noisy/extreme inputs"
```

---

## Task 7: CLI `scenarios` Command

**Files:**
- Create: `src/cli/scenario_cmd.py`
- Modify: `src/main.py`

- [ ] **Step 1: Create `src/cli/scenario_cmd.py`**

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


def _load_matches_df() -> pd.DataFrame:
    data_path = settings.data_dir / "processed" / "matches.csv"
    if not data_path.exists():
        data_path = settings.data_dir / "sample" / "matches.csv"
    return load_matches(data_path)


def cmd_scenarios(home_team: str = "Brazil", away_team: str = "France") -> None:
    """Run scenario analysis + upset check for the given match."""
    from src.features.feature_builder import FeatureBuilder
    from src.ml.xgboost_model import XGBoostMatchModel
    from src.scenarios.scenario_runner import ScenarioRunner, Perturbation
    from src.scenarios.upset_analysis import UpsetAnalyzer

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
        weights = (cfg["elo"], cfg["poisson"], cfg["xgboost"])
    else:
        weights = (settings.ensemble.elo_weight,
                   settings.ensemble.poisson_weight,
                   settings.ensemble.xgboost_weight)

    runner = ScenarioRunner(
        elo_model=elo,
        poisson_model=poisson,
        xgb_model=xgb_model,
        feature_builder=feature_builder,
        weights=weights,
    )

    console.print(Panel(f"Scenario analysis: {home_team} vs {away_team}", style="blue"))

    # Baseline
    baseline_result = runner.run(home_team, away_team, [])
    bp = baseline_result["baseline"]

    # Scenario: away team loses attack strength
    scenario1 = runner.run(
        home_team, away_team,
        [Perturbation(param="attack_strength", team=away_team, delta=-0.3)]
    )
    # Scenario: home team loses recent form
    scenario2 = runner.run(
        home_team, away_team,
        [Perturbation(param="recent_form", team=home_team, delta=-0.2)]
    )

    # Display
    table = Table(
        title=f"Scenario Analysis — {home_team} vs {away_team}",
        show_header=True, header_style="bold cyan"
    )
    table.add_column("Scenario", style="white")
    table.add_column(f"{home_team} Win", style="green", justify="right")
    table.add_column("Draw", style="yellow", justify="right")
    table.add_column(f"{away_team} Win", style="red", justify="right")

    table.add_row("Baseline",
                  f"{bp['home_win']:.1%}", f"{bp['draw']:.1%}", f"{bp['away_win']:.1%}")
    p1 = scenario1["perturbed"]
    table.add_row(f"{away_team} −30% attack",
                  f"{p1['home_win']:.1%}", f"{p1['draw']:.1%}", f"{p1['away_win']:.1%}")
    p2 = scenario2["perturbed"]
    table.add_row(f"{home_team} −20% form",
                  f"{p2['home_win']:.1%}", f"{p2['draw']:.1%}", f"{p2['away_win']:.1%}")

    console.print()
    console.print(table)

    # Upset check
    analyzer = UpsetAnalyzer()
    flag = analyzer.check(
        home_team=home_team,
        away_team=away_team,
        home_win_prob=bp["home_win"],
        away_win_prob=bp["away_win"],
        confidence_band="Medium",
        stability_band="Stable",
    )

    if flag.is_upset:
        console.print("\n[bold yellow]Upset Alert:[/bold yellow]")
        for reason in flag.reasons:
            console.print(f"  • {reason}")
    else:
        console.print("\n[green]No upset potential flagged for baseline prediction.[/green]")

    console.print()
```

- [ ] **Step 2: Wire `scenarios` into `src/main.py`**

Add import:

```python
from src.cli.scenario_cmd import cmd_scenarios
```

Add branch in `main()`:

```python
    elif command == "scenarios":
        home = args[1] if len(args) > 1 else "Brazil"
        away = args[2] if len(args) > 2 else "France"
        cmd_scenarios(home, away)
```

Add usage line:

```python
        console.print("  python main.py scenarios [home_team] [away_team]")
```

- [ ] **Step 3: Smoke-test**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python main.py scenarios Brazil France
```

Expected: table with baseline + two scenario rows, upset check result.

- [ ] **Step 4: Commit**

```bash
git add src/cli/scenario_cmd.py src/main.py
git commit -m "feat: CLI scenarios command — perturbation + upset analysis"
```

---

## Task 8: Reports Package — markdown_reporter.py + forecast_summary.py

**Files:**
- Create: `src/reports/__init__.py`
- Create: `src/reports/markdown_reporter.py`
- Create: `src/reports/forecast_summary.py`
- Test: `tests/test_forecast_summary.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_forecast_summary.py`:

```python
"""Tests for src/reports/forecast_summary.py and markdown_reporter.py."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.reports.markdown_reporter import MarkdownReporter
from src.reports.forecast_summary import ReportSection


class TestMarkdownReporter:
    def test_render_confidence_section(self):
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

    def test_render_drift_section_ok(self):
        reporter = MarkdownReporter()
        drift = {"status": "ok", "n_records": 100, "alerts": []}
        md = reporter.drift_section(drift)
        assert "drift" in md.lower() or "Drift" in md
        assert "ok" in md.lower() or "OK" in md

    def test_render_drift_section_alert(self):
        reporter = MarkdownReporter()
        drift = {
            "status": "alert",
            "n_records": 100,
            "alerts": [
                {"metric": "log_loss", "current_value": 1.2,
                 "baseline_mean": 0.9, "severity": "warning"}
            ],
        }
        md = reporter.drift_section(drift)
        assert "log_loss" in md

    def test_render_returns_string(self):
        reporter = MarkdownReporter()
        md = reporter.confidence_section(
            {"confidence_score": 0.5, "confidence_band": "Medium",
             "factor_breakdown": {}}
        )
        assert isinstance(md, str)
        assert len(md) > 0


class TestForecastSummary:
    def test_generate_report_creates_file(self):
        from src.reports.forecast_summary import generate_report

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

    def test_generate_report_filename_contains_date(self):
        from src.reports.forecast_summary import generate_report

        sections = [ReportSection(title="Test", content="content")]
        with tempfile.TemporaryDirectory() as tmp:
            out_path = generate_report(sections, output_dir=Path(tmp))
            assert "forecast_report" in out_path.name

    def test_report_section_dataclass(self):
        section = ReportSection(title="My Section", content="body text")
        assert section.title == "My Section"
        assert section.content == "body text"
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python -m pytest tests/test_forecast_summary.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.reports'`

- [ ] **Step 3: Create `src/reports/__init__.py`** (empty)

```python
```

- [ ] **Step 4: Create `src/reports/markdown_reporter.py`**

```python
from __future__ import annotations


class MarkdownReporter:
    """Render diagnostic signals and metadata into markdown sections."""

    def confidence_section(self, confidence: dict) -> str:
        score = confidence.get("confidence_score", 0.0)
        band = confidence.get("confidence_band", "Unknown")
        breakdown = confidence.get("factor_breakdown", {})

        lines = [
            "## Confidence Assessment\n",
            f"**Overall Score:** {score:.3f} | **Band:** {band}\n",
        ]

        if breakdown:
            lines.append("| Factor | Score |")
            lines.append("|--------|-------|")
            for factor, value in breakdown.items():
                lines.append(f"| {factor.replace('_', ' ').title()} | {value:.3f} |")

        return "\n".join(lines)

    def drift_section(self, drift: dict) -> str:
        status = drift.get("status", "unknown")
        n = drift.get("n_records", 0)
        alerts = drift.get("alerts", [])

        lines = [
            "## Drift Detection\n",
            f"**Status:** {status.upper()} | **Records analysed:** {n}\n",
        ]

        metrics = drift.get("current_window_metrics", {})
        if metrics:
            lines.append("| Metric | Rolling Value |")
            lines.append("|--------|--------------|")
            for metric, value in metrics.items():
                lines.append(f"| {metric} | {value:.4f} |")
            lines.append("")

        if alerts:
            lines.append("**Alerts:**")
            for alert in alerts:
                lines.append(
                    f"- [{alert['severity'].upper()}] {alert['metric']}: "
                    f"{alert['current_value']:.4f} (baseline {alert['baseline_mean']:.4f})"
                )
        else:
            lines.append("_No drift alerts._")

        return "\n".join(lines)

    def stability_section(self, stability: dict) -> str:
        band = stability.get("stability_band", "Unknown")
        home = stability.get("home_team", "")
        away = stability.get("away_team", "")
        stds = stability.get("std_probabilities", {})

        lines = [
            "## Prediction Stability\n",
            f"**Match:** {home} vs {away}",
            f"**Stability Band:** {band}\n",
        ]

        if stds:
            lines.append("| Outcome | Std Dev |")
            lines.append("|---------|---------|")
            for outcome, std in stds.items():
                lines.append(f"| {outcome.replace('_', ' ').title()} | {std:.4f} |")

        return "\n".join(lines)

    def benchmark_section(self, benchmark_csv_path) -> str:
        try:
            import pandas as pd
            df = pd.read_csv(benchmark_csv_path)
            best_row = df.loc[df["log_loss"].idxmin()]
            lines = [
                "## Model Benchmark\n",
                f"**Best Model:** {best_row['model']} "
                f"(log loss {best_row['log_loss']:.4f})\n",
                "| Model | Accuracy | Log Loss | Brier |",
                "|-------|----------|----------|-------|",
            ]
            for _, row in df.iterrows():
                lines.append(
                    f"| {row['model']} | {row['accuracy']:.4f} | "
                    f"{row['log_loss']:.4f} | {row['brier_score']:.4f} |"
                )
            return "\n".join(lines)
        except Exception as exc:
            return f"## Model Benchmark\n\n_Unavailable: {exc}_"

    def shap_section(self, global_shap: dict | None, top_n: int = 10) -> str:
        if not global_shap:
            return "## Feature Importance (SHAP)\n\n_SHAP not available._"
        sorted_features = sorted(global_shap.items(), key=lambda x: x[1], reverse=True)
        lines = [
            "## Feature Importance (SHAP)\n",
            "| Rank | Feature | Mean |SHAP| |",
            "|------|---------|--------------|",
        ]
        for i, (name, val) in enumerate(sorted_features[:top_n], 1):
            lines.append(f"| {i} | {name} | {val:.6f} |")
        return "\n".join(lines)
```

- [ ] **Step 5: Create `src/reports/forecast_summary.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from loguru import logger


@dataclass
class ReportSection:
    title: str
    content: str


def generate_report(
    sections: list[ReportSection],
    output_dir: Path,
    report_date: date | None = None,
) -> Path:
    """Render sections into a dated markdown file.

    Parameters
    ----------
    sections:
        Ordered list of ReportSection(title, content) to include.
    output_dir:
        Directory to write the report into.
    report_date:
        Date to stamp on the report. Defaults to today.

    Returns
    -------
    Path
        Path of the written markdown file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_date = report_date or date.today()
    filename = f"forecast_report_{report_date.isoformat()}.md"
    out_path = output_dir / filename

    header = (
        f"# Forecast Reliability Report\n\n"
        f"**Generated:** {report_date.isoformat()}\n\n---\n\n"
    )

    body_parts = [f"## {s.title}\n\n{s.content}\n" for s in sections]
    full_text = header + "\n".join(body_parts)

    out_path.write_text(full_text)
    logger.info(f"Forecast report written to {out_path}")
    return out_path
```

- [ ] **Step 6: Run report tests**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python -m pytest tests/test_forecast_summary.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/reports/__init__.py src/reports/markdown_reporter.py src/reports/forecast_summary.py tests/test_forecast_summary.py
git commit -m "feat: reports package — MarkdownReporter + generate_report"
```

---

## Task 9: CLI `report` Command

**Files:**
- Create: `src/cli/report_cmd.py`
- Modify: `src/main.py`

- [ ] **Step 1: Create `src/cli/report_cmd.py`**

```python
from __future__ import annotations

import sys
from datetime import date

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from src.config.settings import settings
from src.ingestion.match_loader import load_matches
from src.models.elo_model import EloModel
from src.models.poisson_model import PoissonModel
from src.reports.markdown_reporter import MarkdownReporter
from src.reports.forecast_summary import ReportSection, generate_report

console = Console()


def _load_matches_df() -> pd.DataFrame:
    data_path = settings.data_dir / "processed" / "matches.csv"
    if not data_path.exists():
        data_path = settings.data_dir / "sample" / "matches.csv"
    return load_matches(data_path)


def cmd_report() -> None:
    """Generate a full forecast reliability report to outputs/reports/."""
    from src.diagnostics.confidence_scorer import ConfidenceScorer
    from src.diagnostics.drift_detection import DriftDetector
    from src.diagnostics.prediction_stability import StabilityAnalyzer
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

    console.print(Panel("Generating forecast reliability report…", style="blue"))

    reporter = MarkdownReporter()
    sections: list[ReportSection] = []

    # --- Benchmark section ---
    benchmark_csv = settings.outputs_dir / "evaluation" / "benchmark_results.csv"
    bm_md = reporter.benchmark_section(benchmark_csv)
    sections.append(ReportSection(title="Model Benchmark", content=bm_md))

    # --- Confidence + drift via test-set predictions ---
    ece = 0.05  # fallback
    if benchmark_csv.exists():
        try:
            import pandas as pd_inner
            df = pd_inner.read_csv(benchmark_csv)
            row = df[df["model"].str.contains("Ensemble", case=False, na=False)]
            if not row.empty:
                ece = float(row["ece"].iloc[0])
        except Exception:
            pass

    conf_scorer = ConfidenceScorer(matches_df=matches_df, calibration_ece=ece)
    detector = DriftDetector()

    test_matches = matches_df[
        pd.to_datetime(matches_df["date"]) >= pd.Timestamp(settings.ml.test_split_date)
    ].sort_values("date").head(200)  # cap for speed

    sample_confidence = None
    for _, row in test_matches.iterrows():
        home, away = str(row["home_team"]), str(row["away_team"])
        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        outcome = "H" if hg > ag else ("D" if hg == ag else "A")
        try:
            result = ensemble.predict(home, away, pd.Timestamp(row["date"]))
            probs = result["ensemble_probabilities"]
            detector.add_record(outcome, probs)
            if sample_confidence is None:
                features = feature_builder.build_features_for_match(
                    home, away, pd.Timestamp(row["date"])
                )
                sample_confidence = conf_scorer.score(
                    home, away,
                    result["component_models"],
                    probs,
                    features,
                )
        except Exception as exc:
            logger.warning(f"Skipping {home} vs {away}: {exc}")

    if sample_confidence:
        sections.append(
            ReportSection(
                title="Confidence Assessment (Sample)",
                content=reporter.confidence_section(sample_confidence),
            )
        )

    drift_status = detector.status()
    sections.append(
        ReportSection(title="Drift Detection", content=reporter.drift_section(drift_status))
    )

    # --- SHAP section (optional) ---
    from src.explainability.shap_engine import _SHAP_AVAILABLE
    if _SHAP_AVAILABLE:
        try:
            from src.explainability.shap_engine import SHAPEngine
            shap_engine = SHAPEngine(xgb_model=xgb_model)
            # Build a small test set for global SHAP
            X_test_rows = []
            for _, row in test_matches.head(50).iterrows():
                try:
                    feat = feature_builder.build_features_for_match(
                        str(row["home_team"]), str(row["away_team"]),
                        pd.Timestamp(row["date"])
                    )
                    X_test_rows.append(feat)
                except Exception:
                    pass
            if X_test_rows:
                import pandas as pd_inner2
                from src.features.feature_builder import FEATURE_COLUMNS
                X_test = pd_inner2.DataFrame(X_test_rows)[FEATURE_COLUMNS]
                global_shap = shap_engine.global_shap(X_test)
                sections.append(
                    ReportSection(
                        title="Feature Importance (SHAP)",
                        content=reporter.shap_section(global_shap),
                    )
                )
        except Exception as exc:
            logger.warning(f"SHAP section skipped: {exc}")
            sections.append(
                ReportSection(title="Feature Importance (SHAP)", content="_SHAP unavailable._")
            )
    else:
        sections.append(
            ReportSection(title="Feature Importance (SHAP)", content="_Install shap for this section._")
        )

    out_dir = settings.outputs_dir / "reports"
    out_path = generate_report(sections, output_dir=out_dir)

    console.print(f"\n[green]Report written to:[/green] {out_path}\n")
```

- [ ] **Step 2: Wire `report` into `src/main.py`**

Add import:

```python
from src.cli.report_cmd import cmd_report
```

Add branch in `main()`:

```python
    elif command == "report":
        cmd_report()
```

Add usage line:

```python
        console.print("  python main.py report")
```

- [ ] **Step 3: Smoke-test**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python main.py report
```

Expected: prints path to generated report file; `outputs/reports/forecast_report_YYYY-MM-DD.md` exists.

- [ ] **Step 4: Confirm report file contains expected sections**

```bash
cd /Users/edwardchiang/2026-world && grep "##" outputs/reports/forecast_report_*.md | head -20
```

Expected: headings for Benchmark, Confidence, Drift, Feature Importance.

- [ ] **Step 5: Commit**

```bash
git add src/cli/report_cmd.py src/main.py
git commit -m "feat: CLI report command — auto-generated forecast reliability report"
```

---

## Task 10: Full Test Suite + Coverage Check

**Files:** none new

- [ ] **Step 1: Run full test suite**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python -m pytest tests/ -q --tb=short
```

Expected: all tests pass (original 200 + new tests from Phase 3a and 3b).

- [ ] **Step 2: Check coverage on new Phase 3 modules**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python -m pytest --cov=src tests/ --cov-report=term-missing -q 2>&1 | grep -E "optimization|diagnostics|explainability|scenarios|reports|TOTAL"
```

Expected: each new package at 80%+, TOTAL at 85%+.

- [ ] **Step 3: Run all six new CLI commands in sequence**

```bash
cd /Users/edwardchiang/2026-world && \
  .venv/bin/python main.py optimize && \
  .venv/bin/python main.py confidence Brazil France && \
  .venv/bin/python main.py drift_check && \
  .venv/bin/python main.py explain Brazil France && \
  .venv/bin/python main.py scenarios Brazil France && \
  .venv/bin/python main.py report
```

Expected: each command completes without error. `explain` prints the graceful "unavailable" message if shap is not installed.

---

## Task 11: Phase 3b Wrap-Up — CLAUDE.md + Tag

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update `CLAUDE.md` — add Phase 3b modules to Module Responsibilities**

In the `## Module Responsibilities` section, add after the Phase 3a section:

```markdown
### Phase 3b Modules

- `explainability/shap_engine.py` — `SHAPEngine`. Optional-dep wrapper around `shap.TreeExplainer`. Guards import with `_SHAP_AVAILABLE` flag. Raises `ImportError` with install hint if shap is missing. Provides `global_shap(X)` and `local_shap(X, outcome)`.
- `explainability/feature_impact.py` — `FeatureImpactRanker`. Wraps `SHAPEngine` to rank features by mean |SHAP| and format as markdown.
- `explainability/prediction_explainer.py` — `PredictionExplainer`. Formats local SHAP into per-match "top drivers" dicts and saves as .md + .json to `outputs/reports/explanations/`.
- `scenarios/scenario_runner.py` — `ScenarioRunner`. Applies named `Perturbation` objects (attack_strength, defense_strength, elo_rating, recent_form) and returns probability delta tables.
- `scenarios/upset_analysis.py` — `UpsetAnalyzer`. Flags matches where underdog win prob > threshold, or confidence is Low, or stability is Unstable.
- `scenarios/stress_tests.py` — `RobustnessTester`. Stress-tests XGBoost against missing/noisy/sparse/extreme feature inputs. Verifies no uncaught exceptions and probabilities remain in [0,1].
- `reports/markdown_reporter.py` — `MarkdownReporter`. Renders confidence, drift, stability, benchmark, and SHAP into markdown sections.
- `reports/forecast_summary.py` — `generate_report(sections, output_dir)`. Assembles `ReportSection` list into a dated `forecast_report_YYYY-MM-DD.md`.
- `cli/explain_cmd.py` — `cmd_explain()`. Calls SHAPEngine with graceful degradation.
- `cli/scenario_cmd.py` — `cmd_scenarios()`. Runs baseline + two scenario perturbations + upset check.
- `cli/report_cmd.py` — `cmd_report()`. Orchestrates all Phase 3 signals into a full report.
```

- [ ] **Step 2: Update Architecture table in CLAUDE.md**

Add rows:

```markdown
| Explainability | `src/explainability/` | ml, features (shap optional) |
| Scenarios | `src/scenarios/` | models, ensemble, features, diagnostics |
| Reports | `src/reports/` | diagnostics, evaluation |
```

- [ ] **Step 3: Run full test suite one final time**

```bash
cd /Users/edwardchiang/2026-world && .venv/bin/python -m pytest tests/ -q --tb=short
```

Expected: all tests pass.

- [ ] **Step 4: Phase 3b commit + tag**

```bash
git add CLAUDE.md
git commit -m "docs: CLAUDE.md updated for Phase 3b — explainability, scenarios, reports complete"
git tag v0.3b-phase3b-explainability-scenarios
```

---

*Phase 3 complete. System is now a serious probabilistic forecasting platform with optimised weights, meaningful confidence scores, drift detection, optional SHAP explanations, scenario analysis, and analyst-grade reports.*
