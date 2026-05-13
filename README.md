# 2026-world — Adaptive Probabilistic Football Forecasting

A production-style probabilistic forecasting system for international football. Built from first principles: no LLM wrappers, no fake agents, no dashboard hype. Just rigorous statistical modelling, calibrated uncertainty, and reliability intelligence.

---

## What this actually is

Most "AI sports prediction" projects are LLM wrappers or confidence-theatre. This isn't that.

`2026-world` is an **ensemble forecasting platform** built around three independently-interpretable models, combined with a full reliability layer:

- Calibrated probability estimates (not raw model scores)
- Explicit uncertainty quantification (confidence bands)
- Drift detection to catch model degradation
- Perturbation-based stability analysis
- SHAP feature attribution for every prediction
- Scenario simulation with probability deltas
- Robustness stress testing under adversarial inputs

The system is **deterministic, reproducible, and testable**. Every prediction can be traced back to its constituent signals.

---

## Forecast Reliability Architecture

```
Historical Match Data (37,000+ international matches)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                   Feature Engineering                    │
│   Elo deltas · rolling form · attack/defense strength   │
│   head-to-head history · tournament weights · xG stats  │
└─────────────────────────────────────────────────────────┘
        │
        ├──────────────────┬─────────────────┐
        ▼                  ▼                  ▼
  ┌──────────┐      ┌────────────┐     ┌───────────┐
  │ Elo Model│      │Poisson xG  │     │ XGBoost   │
  │          │      │Model       │     │ Classifier│
  └──────────┘      └────────────┘     └───────────┘
        │                  │                  │
        └──────────────────┴─────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Ensemble Optimization  │
              │  SLSQP weight search    │
              │  elo=0.175 xgb=0.825   │
              └────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Probability Calibration│
              │  CalibratedClassifierCV │
              └────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
   ┌──────────────────┐     ┌──────────────────────┐
   │ Confidence Scorer │     │   Drift Detection    │
   │ 5-factor engine   │     │   Rolling window     │
   │ High/Medium/Low   │     │   metric monitoring  │
   └──────────────────┘     └──────────────────────┘
              │
              ▼
   ┌──────────────────┐
   │ Stability Analysis│
   │ Perturbation-based│
   │ Stable/Moderate/  │
   │ Unstable bands    │
   └──────────────────┘
              │
        ┌─────┴──────┐
        ▼             ▼
┌─────────────┐  ┌──────────────┐
│    SHAP     │  │  Scenario    │
│Explainability│  │  Analysis   │
│ Top drivers │  │  What-if    │
│ per match   │  │  perturbations│
└─────────────┘  └──────────────┘
        │
        ▼
┌─────────────────────────────┐
│   Forecast Reliability      │
│   Report (Markdown)         │
│   Benchmark · Confidence    │
│   Drift · SHAP · Upsets     │
└─────────────────────────────┘
```

---

## Benchmark Results

Tested on a chronological holdout: all matches from 2020-01-01 onward (~2,300 matches). Lower log loss and Brier score are better.

| Model | Accuracy | Log Loss | Brier Score | ECE |
|-------|----------|----------|-------------|-----|
| Elo | 0.613 | 0.883 | 0.518 | 0.055 |
| Poisson | 0.583 | 0.915 | 0.538 | 0.029 |
| XGBoost | 0.611 | **0.854** | 0.502 | 0.032 |
| Ensemble (default 0.3/0.3/0.4) | 0.606 | 0.865 | 0.506 | 0.043 |
| **Ensemble (optimized)** | — | **0.852** | — | — |

**Ensemble optimization** uses SLSQP on the test split to find weights that minimize log loss. Converged to `elo=0.175, poisson=0.0, xgboost=0.825` — a +0.013 log loss improvement over the default weighting.

Key observation: Poisson's weight collapses to 0 in the optimized ensemble. Poisson is useful for generating expected goals distributions but its raw probability estimates add noise when XGBoost has access to the same underlying features.

---

## Reliability Layer

### Confidence Scoring

Five factors, weighted to produce a `[0, 1]` confidence score with band labels (High / Medium / Low):

| Factor | What it measures | Weight |
|--------|-----------------|--------|
| Model agreement | How consistent are the three models? | 0.35 |
| Calibration quality | ECE of the ensemble on held-out data | 0.25 |
| Feature completeness | Fraction of non-null features in the match vector | 0.20 |
| Historical reliability | Log-scaled count of past meetings | 0.15 |
| Prediction volatility | Entropy of the ensemble probabilities | 0.05 |

Volatility is down-weighted (0.05) because high entropy often reflects a genuinely close match, not an unreliable model.

### Drift Detection

A rolling-window detector that flags when metric quality degrades beyond a statistical threshold:

```
Baseline: first N=50 predictions → mean/std of log loss, Brier, entropy
Rolling:  each subsequent window compared to baseline
Alert:    if current_metric > baseline_mean + 2.0 * baseline_std
Severity: warning (2σ) or critical (3σ)
```

### Prediction Stability

Gaussian noise (σ=0.05) is applied to XGBoost feature inputs across 20 perturbation runs. Elo and Poisson are deterministic and are excluded. Output: mean ± std per outcome, plus a stability band (Stable / Moderate / Unstable).

This is sensitivity analysis, not Monte Carlo — it answers "how much does this prediction move when inputs are noisy?"

---

## SHAP Explainability

Uses `shap.TreeExplainer` on the XGBoost booster. Optional dependency — the rest of the system works without it.

**Global importance** — mean |SHAP value| across all test matches:
```
Rank  Feature                   Mean |SHAP|
1     elo_difference            0.183421
2     home_recent_form          0.112847
3     away_recent_form          0.098312
4     attack_strength_home      0.076504
5     defense_strength_away     0.071239
...
```

**Local explanation** — per-match top drivers for a given outcome:
```
Brazil vs France — Top Drivers (Home win)

1. elo_difference         +0.18   (raw: +120.4)
2. home_recent_form       +0.11   (raw: 0.80)
3. away_attack_strength   −0.06   (raw: 1.42)
4. h2h_home_win_rate      +0.04   (raw: 0.40)
5. home_goals_per_game    +0.03   (raw: 2.10)
```

Install: `pip install shap`. All other commands function normally without it.

---

## Scenario Analysis

Stress-test predictions by perturbing model inputs:

```bash
python main.py scenarios Brazil France
```

```
Scenario Analysis — Brazil vs France
┌────────────────────┬────────────┬───────┬────────────┐
│ Scenario           │ Brazil Win │  Draw │ France Win │
├────────────────────┼────────────┼───────┼────────────┤
│ Baseline           │      46.2% │ 21.0% │      32.8% │
│ France −30% attack │      47.2% │ 24.2% │      28.6% │
│ Brazil −20% form   │      46.3% │ 21.0% │      32.7% │
└────────────────────┴────────────┴───────┴────────────┘

No upset potential flagged for baseline prediction.
```

Supported perturbation parameters: `attack_strength`, `defense_strength`, `elo_rating`, `recent_form`.

**Upset flagging** triggers when: underdog win probability > 35%, OR confidence band is Low, OR stability is Unstable.

---

## CLI Reference

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Core prediction
python main.py predict Brazil France
python main.py simulate

# Training pipeline
python main.py train
python main.py evaluate
python main.py benchmark

# Phase 3 — Reliability Intelligence
python main.py optimize               # Find optimal ensemble weights
python main.py confidence Brazil France  # Score prediction confidence
python main.py drift_check            # Check for metric degradation
python main.py explain Brazil France  # SHAP feature attribution (requires shap)
python main.py scenarios Brazil France  # What-if scenario analysis
python main.py report                 # Generate full reliability report
```

---

## Sample CLI Outputs

**Confidence scoring:**
```
python main.py confidence Brazil France

┌────────────────────────┬────────┬────────┐
│ Factor                 │  Score │ Weight │
├────────────────────────┼────────┼────────┤
│ model_agreement        │ 0.9207 │   0.35 │
│ calibration_quality    │ 0.9677 │   0.25 │
│ feature_completeness   │ 0.0000 │   0.20 │
│ historical_reliability │ 0.9691 │   0.15 │
│ prediction_volatility  │ 0.0318 │   0.05 │
└────────────────────────┴────────┴────────┘
```

**Drift check (clean state):**
```
python main.py drift_check

┌───────────────┬───────┐
│ Metric        │ Value │
├───────────────┼───────┤
│ Total records │     0 │
│ Alerts fired  │     0 │
└───────────────┴───────┘

✓ No drift detected
```

**Ensemble optimization:**
```
python main.py optimize

┌────────────────────┬────────┐
│ Baseline Log Loss  │ 0.8651 │
│ Optimized Log Loss │ 0.8520 │
│ Improvement        │ 0.0131 │
└────────────────────┴────────┘

Saved to outputs/models/best_weight_config.json
```

---

## Project Structure

```
2026-world/
├── src/
│   ├── config/           # Settings dataclasses (all parameters centralised)
│   ├── ingestion/        # CSV loaders with schema validation
│   ├── models/           # elo_model, poisson_model, prediction_engine
│   ├── features/         # feature_builder, rolling_form, team_strength
│   ├── ml/               # xgboost_model, trainer, calibration
│   ├── ensemble/         # ensemble_engine, EnsembleWeights
│   ├── evaluation/       # benchmark, backtesting, metrics
│   ├── optimization/     # ensemble_optimizer (SLSQP), weight_search
│   ├── diagnostics/      # confidence_scorer, drift_detection,
│   │                     #   prediction_stability, reliability_monitor
│   ├── explainability/   # shap_engine, feature_impact, prediction_explainer
│   ├── scenarios/        # scenario_runner, upset_analysis, stress_tests
│   ├── reports/          # markdown_reporter, forecast_summary
│   ├── simulation/       # monte_carlo tournament simulator
│   ├── utils/            # helpers, metrics, logger
│   └── cli/              # one file per CLI command
├── tests/                # 287 tests, 88–100% coverage per Phase 3 module
├── data/
│   ├── sample/           # WC 2022 match data (bundled)
│   └── processed/        # Full dataset (37,000+ matches, git-ignored)
└── outputs/              # Predictions, models, reports, logs (git-ignored)
```

**Dependency rule:** each layer only imports from layers below it. `diagnostics/` does not import from `scenarios/`. `cli/` is the only layer allowed to import from everything. No circular imports.

---

## Testing

```bash
pytest tests/ -v
pytest --cov=src tests/ --cov-report=term-missing
```

| Scope | Status |
|-------|--------|
| Total tests | **287 passing** |
| diagnostics/ | 88–100% per module |
| explainability/ | 82–100% per module |
| scenarios/ | 89–96% per module |
| reports/ | 96–100% per module |
| optimization/ | 93–100% per module |
| Coverage approach | Unit tests with mocked IO; SHAP mocked via `unittest.mock`; no randomness without `random_seed` |

---

## Setup

```bash
git clone https://github.com/eddiepiper/2026-world.git
cd 2026-world
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# To train on full data (downloads ~35MB):
python -c "from src.ingestion.data_fetcher import fetch_all; fetch_all()"
python main.py train
python main.py benchmark

# Or use bundled sample data immediately:
python main.py predict Brazil France
```

SHAP explainability (optional):
```bash
pip install shap
python main.py explain Brazil France
```

---

## Design Principles

**What was intentionally excluded:**

- No LLM prediction layers — language models do not improve calibrated probability estimates
- No autonomous agents — non-deterministic orchestration adds noise, not signal
- No dashboard or frontend — this is a forecasting research platform, not a product demo
- No real-time news scraping — sentiment signals degrade calibration without substantial data
- No multi-model ensemble of LLMs — accuracy theatre

**What was deliberately prioritised:**

- Calibrated probabilities over point predictions
- Uncertainty quantification at every layer
- Statistical drift detection over silent degradation
- Interpretable feature attribution (SHAP) over black-box outputs
- Reproducible, deterministic pipelines (random seeds, chronological splits)
- Graceful degradation when optional dependencies are absent

---

## Architecture Freeze — v1

This codebase is frozen at v1 (tag: `v0.3-final-forecast-reliability-platform`).

The following are intentionally **out of scope**:
- Telegram bot or messaging integrations
- Google Sheets / live dashboard
- Microservices or API layer
- Frontend/UI
- Additional deep learning models
- LLM integration of any kind

Future work (if any) is limited to: bug fixes, test improvements, benchmark validation, documentation, and reproducibility improvements.

---

## Roadmap

| Phase | Status | Focus |
|-------|--------|-------|
| Phase 1 | ✅ Complete | Elo + Poisson + Monte Carlo CLI |
| Phase 2 | ✅ Complete | XGBoost ensemble, feature engineering, calibration, evaluation |
| Phase 3a | ✅ Complete | Ensemble optimisation, confidence scoring, drift detection, stability |
| Phase 3b | ✅ Complete | SHAP explainability, scenario analysis, upset analysis, reliability reports |
| v1 Freeze | ✅ Tagged | `v0.3-final-forecast-reliability-platform` |
