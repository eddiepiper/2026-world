# Phase 3 Design — Forecasting Intelligence Layer

**Date:** 2026-05-12
**Project:** 2026-world
**Phase:** 3a (optimize + confidence + drift + stability) then 3b (SHAP + scenarios + reports + robustness)
**Status:** Approved

---

## 1. Objective

Move from "model predicts" to "forecasting system understands uncertainty and reliability."

Phase 2 established a working ML ensemble. Phase 3 makes it trustworthy:
- Fix the ensemble underperforming XGBoost (optimize weights)
- Produce meaningful confidence scores (not fake certainty)
- Detect when the system becomes unreliable (drift detection)
- Explain predictions with evidence (SHAP)
- Stress-test assumptions (scenarios + robustness)
- Generate analyst-grade reports

This phase is NOT about: UI, Telegram, APIs, dashboards, news scraping.

---

## 2. Current Baseline

From Phase 2 benchmark (2020–2026 test set):

| Model | Log Loss | Brier | Accuracy |
|---|---|---|---|
| XGBoost | **0.8544** | 0.5015 | 0.6107 |
| Ensemble (0.3/0.3/0.4) | 0.8651 | 0.5064 | 0.6063 |
| Elo | 0.8825 | 0.5181 | 0.6130 |
| Poisson | 0.9152 | 0.5381 | 0.5827 |

**Problem:** Ensemble underperforms XGBoost because fixed weights (0.3/0.3/0.4) are not optimal. Task 1 fixes this.

---

## 3. Scope Split

### Phase 3a
1. Ensemble optimization
2. Confidence scoring
3. Drift detection
4. Prediction stability

### Phase 3b
5. SHAP explainability
6. Scenario analysis
7. Upset analysis
8. Forecast reliability reports
9. Robustness testing

Each sub-phase is independently committed and validated.

---

## 4. Package Structure

```
src/
├── optimization/
│   ├── __init__.py
│   ├── ensemble_optimizer.py   # scipy L-BFGS-B weight search
│   └── weight_search.py        # grid search + weight validation helpers
│
├── diagnostics/
│   ├── __init__.py
│   ├── confidence_scorer.py    # 5-factor confidence engine
│   ├── drift_detection.py      # rolling metric windows + drift alerts
│   ├── prediction_stability.py # perturbation-based variance analysis
│   └── reliability_monitor.py  # aggregates all diagnostic signals
│
├── explainability/
│   ├── __init__.py
│   ├── shap_engine.py          # TreeExplainer wrapper (global + local)
│   ├── feature_impact.py       # global feature importance ranking
│   └── prediction_explainer.py # per-match "top drivers" formatter
│
├── scenarios/
│   ├── __init__.py
│   ├── scenario_runner.py      # apply perturbations, re-run ensemble
│   ├── upset_analysis.py       # flag high-variance / overvalued favorites
│   └── stress_tests.py         # missing features, noisy inputs, low-data teams
│
├── reports/
│   ├── __init__.py
│   ├── markdown_reporter.py    # render diagnostic signals to .md
│   └── forecast_summary.py     # top-level forecast_report() entry point
│
└── cli/
    ├── __init__.py
    ├── optimize_cmd.py
    ├── diagnostic_cmd.py
    ├── explain_cmd.py
    ├── scenario_cmd.py
    └── report_cmd.py
```

---

## 5. Dependency Graph

```
config → utils → ingestion → models → features → ml → ensemble → evaluation
                                                                      ↓
                                                optimization   diagnostics   explainability
                                                      ↓              ↓
                                                   scenarios → reports
                                                      ↑
                                                diagnostics
```

**Rule:** No layer imports from above it. Phase 3 packages do not form cycles.
- `scenarios/` may import from `diagnostics/` (for confidence scores)
- `reports/` imports from `diagnostics/` and `optimization/`
- `explainability/` is standalone — imports only from `ml/` and `features/`

---

## 6. Task Designs

### Task 1 — Ensemble Optimizer

**File:** `src/optimization/ensemble_optimizer.py`

Algorithm:
1. Load the Phase 2 benchmark test set (matches after `test_split_date`)
2. For each match, get raw Elo, Poisson, XGBoost probabilities
3. Use `scipy.optimize.minimize` (L-BFGS-B) to find weights minimizing log loss
4. Constrain: weights ≥ 0, weights sum = 1 (3-simplex constraint)
5. Save `outputs/models/best_weight_config.json`

```python
# Constraint: weights sum to 1
constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
bounds = [(0, 1), (0, 1), (0, 1)]
result = scipy.optimize.minimize(objective, x0=[0.3, 0.3, 0.4],
                                  constraints=constraints, bounds=bounds,
                                  method="SLSQP")
```

Note: L-BFGS-B doesn't support equality constraints — use SLSQP instead.

Output file format:
```json
{
  "elo": 0.18,
  "poisson": 0.22,
  "xgboost": 0.60,
  "optimized_log_loss": 0.841,
  "baseline_log_loss": 0.865,
  "improvement": 0.024,
  "method": "SLSQP",
  "validated_at": "2026-05-12"
}
```

CLI: `python main.py optimize`

### Task 2 — Confidence Scorer

**File:** `src/diagnostics/confidence_scorer.py`

Five factors, weighted mean, output normalized to [0, 1]:

| Factor | Computation | Weight |
|---|---|---|
| Model agreement | 1 − mean pairwise max-diff across outcomes | 0.35 |
| Calibration quality | 1 − ECE from benchmark (model-level, from benchmark_results.csv) | 0.25 |
| Feature completeness | fraction of non-null features in match feature vector | 0.20 |
| Historical reliability | log(1 + match_count(home, away)) / log(1 + max_count) | 0.15 |
| Prediction volatility | 1 − entropy(ensemble_probs) / log(3) | 0.05 |

**Design note on volatility:** High entropy often indicates a genuinely close match, not an unreliable forecast. Volatility is intentionally down-weighted (0.05) to avoid penalizing well-calibrated uncertainty.

Bands:
- Low: score < 0.40
- Medium: 0.40 ≤ score < 0.70
- High: score ≥ 0.70

Output:
```python
{
  "confidence_score": 0.81,
  "confidence_band": "High",
  "factor_breakdown": {
    "model_agreement": 0.88,
    "calibration_quality": 0.82,
    "feature_completeness": 1.00,
    "historical_reliability": 0.65,
    "prediction_volatility": 0.72,
  }
}
```

CLI: `python main.py confidence Brazil France`

### Task 3 — Drift Detection

**File:** `src/diagnostics/drift_detection.py`

Configuration: add `DiagnosticsConfig` to `settings.py`:
```python
@dataclass
class DiagnosticsConfig:
    drift_window: int = 50         # matches per rolling window
    drift_alert_threshold: float = 2.0  # std deviations above baseline
    stability_n_perturbations: int = 20
    stability_noise_scale: float = 0.05
```

Algorithm:
1. Accept a list of (prediction, outcome) pairs
2. Compute baseline metrics from the first `drift_window` pairs
3. For each subsequent window (step=1), compute rolling log loss, Brier score, prediction entropy
4. Alert if any metric exceeds `baseline_mean + threshold * baseline_std`

`DriftAlert` dataclass:
```python
@dataclass
class DriftAlert:
    metric: str          # "log_loss" | "brier_score" | "entropy"
    current_value: float
    baseline_mean: float
    baseline_std: float
    severity: str        # "warning" | "critical"
```

CLI: `python main.py drift_check`

### Task 4 — Prediction Stability

**File:** `src/diagnostics/prediction_stability.py`

For a given match, run ensemble `stability_n_perturbations` times with ±`stability_noise_scale` Gaussian noise on continuous features. Report:
- Mean probability per outcome
- Standard deviation per outcome
- Stability band: Stable (std < 0.02), Moderate (0.02–0.05), Unstable (> 0.05)

This is NOT a Monte Carlo simulation — it's sensitivity analysis on the feature vector.

### Task 5 — SHAP Explainability

**File:** `src/explainability/shap_engine.py`

Dependency: `shap` (optional — graceful degradation if unavailable)

If `shap` is not installed or no `model.pkl` exists, `explain` and any SHAP-dependent report sections fail with a clear user-facing message (e.g., `"Explainability unavailable: install shap via pip install shap"`) and exit cleanly. No stack traces. Do NOT add shap to `requirements.txt`.

Uses `shap.TreeExplainer(xgb_model.model)`. Two modes:
- **Global**: `shap_values = explainer.shap_values(X_test)` → mean |SHAP| per feature
- **Local**: `shap_values = explainer.shap_values(X_single)` → per-match top drivers

`prediction_explainer.py` formats local output:
```
Brazil vs France — Top drivers (home win):
1. elo_difference         +0.18
2. home_recent_form       +0.11
3. away_attack_strength   −0.06
```

Saves to:
- `outputs/reports/explanations/<home>_vs_<away>_<date>.md`
- `outputs/reports/explanations/<home>_vs_<away>_<date>.json`

CLI: `python main.py explain Brazil France`

### Task 6 — Scenario Analysis

**File:** `src/scenarios/scenario_runner.py`

Perturbation dict:
```python
{"param": "attack_strength", "team": "France", "delta": -0.3}
```

Supported params: `attack_strength`, `defense_strength`, `elo_rating`, `recent_form`

Output: probability delta table comparing original vs perturbed predictions.

### Task 7 — Upset Analysis

**File:** `src/scenarios/upset_analysis.py`

Flag matches where:
- Underdog win probability > 35%, OR
- Confidence band is "Low", OR
- Prediction stability is "Unstable"

### Task 8 — Forecast Reports

**File:** `src/reports/forecast_summary.py`

Generates `outputs/reports/forecast_report_YYYY-MM-DD.md` with sections:
- Best performing model (from benchmark)
- Calibration quality summary
- Confidence distribution across recent predictions
- Drift status (alert or clear)
- Top 10 global SHAP features
- Recent upset flags

CLI: `python main.py report`

### Task 9 — Robustness Testing

**File:** `src/scenarios/stress_tests.py`

Test scenarios:
- Missing features: zero out or NaN each feature column one at a time
- Noisy inputs: ±20% noise on all continuous features simultaneously
- Low-data teams: teams with < 10 historical matches
- Calibration edge cases: extreme probabilities (0.01, 0.99)

Goal: system degrades gracefully, no uncaught exceptions.

---

## 7. CLI Expansion

New commands implemented in `src/cli/*.py` files; `src/main.py` remains the dispatcher and imports from `src/cli/`:

```
python main.py optimize              # Task 1 — find optimal weights
python main.py confidence <H> <A>   # Task 2 — confidence score for a match
python main.py drift_check          # Task 3 — check for prediction drift
python main.py explain <H> <A>      # Task 5 — SHAP explanation for a match
python main.py scenarios            # Task 6+7 — scenario + upset analysis
python main.py report               # Task 8 — generate forecast report
```

---

## 8. Settings Changes

Add to `src/config/settings.py`:

```python
@dataclass
class DiagnosticsConfig:
    drift_window: int = 50
    drift_alert_threshold: float = 2.0
    stability_n_perturbations: int = 20
    stability_noise_scale: float = 0.05

@dataclass
class Settings:
    ...
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
```

---

## 9. New Dependencies

No new hard dependencies. `shap` is optional:

```
# Optional — required only for 'explain' command
# pip install shap
```

`shap_engine.py` guards the import at the top of the file:
```python
try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
```

All public functions in `shap_engine.py` check `_SHAP_AVAILABLE` at entry and raise a descriptive `ImportError` if missing. CLI commands catch this and print a clean message instead of a traceback.

---

## 10. Testing Requirements

Target: 90%+ coverage across all Phase 3 `src/` code.

Test files to create:
- `tests/test_ensemble_optimizer.py`
- `tests/test_confidence_scorer.py`
- `tests/test_drift_detection.py`
- `tests/test_prediction_stability.py`
- `tests/test_shap_engine.py` (mock `shap.TreeExplainer`; also test graceful failure when `_SHAP_AVAILABLE = False`)
- `tests/test_scenario_runner.py`
- `tests/test_forecast_summary.py`

Testing notes:
- Mock `shap.TreeExplainer` in SHAP tests — do not call real SHAP on small fixtures
- Use deterministic fixtures with `random_seed` everywhere
- Drift detection tests must provide enough pairs to fill at least one window

---

## 11. Engineering Rules (Carry-Forward from Phase 2)

- No leakage: optimization uses same chronological test split as benchmark
- Deterministic: set `random_seed` in all stochastic operations
- No hardcoded probabilities
- No fake confidence scores (every factor must be computed from real data)
- Fail gracefully on missing features, unknown teams, insufficient history
- All paths via `pathlib.Path`
- Log with `loguru`, not `print`
- No circular imports

---

## 12. Success Criteria

At the end of Phase 3:

1. Optimized ensemble outperforms XGBoost on log loss
2. Confidence scores correlate with actual prediction accuracy
3. Drift detection identifies degradation windows in backtesting
4. SHAP explanations render correctly for any trained model
5. Scenario analysis produces meaningful probability delta tables
6. Forecast reliability reports generate without errors
7. All new code covered at 90%+ with tests passing
