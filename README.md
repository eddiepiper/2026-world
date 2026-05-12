# 2026-world — World Cup AI Prediction Engine

A modular, production-grade AI system for football match prediction, tournament simulation, and football intelligence research. Built as an AI engineering portfolio project.

---

## What it does

- **Elo Rating Engine** — dynamic team ratings updated from historical match data
- **Poisson xG Model** — attack/defense strength + home advantage → expected goals
- **Match Prediction** — blended Elo + Poisson probabilities with configurable weights
- **Monte Carlo Simulation** — 10,000-run tournament bracket → winner/finalist/semifinal probabilities
- **XGBoost Ensemble** — machine learning classifier trained on engineered features, blended with Elo + Poisson
- **Evaluation & Benchmarking** — accuracy, log loss, Brier score compared across all models

---

## Architecture

### Phase 1 — Elo + Poisson + Monte Carlo

```
2026-world/
├── src/
│   ├── config/          # Settings dataclasses (EloConfig, PoissonConfig, ...)
│   ├── ingestion/       # CSV loaders with validation
│   ├── models/          # elo_model, poisson_model, prediction_engine
│   ├── simulation/      # monte_carlo tournament simulator
│   └── utils/           # logger, helpers, metrics
├── data/sample/         # WC 2022 match data, teams, FIFA rankings
├── tests/               # Unit tests (Elo, Poisson, simulation)
└── outputs/             # Predictions, simulations, logs (git-ignored)
```

### Phase 2 — ML Ensemble

```
src/
├── features/            # feature_builder, rolling_form, team_strength
│   └─ builds 20+ engineered features per match (Elo deltas, form, xG, etc.)
├── ml/                  # xgboost_model, trainer, calibration
│   └─ XGBoost multiclass classifier (H/D/A) with chronological train/test split
├── ensemble/            # ensemble_engine
│   └─ blends Elo + Poisson + XGBoost (configurable weights) → confidence score
└── evaluation/          # benchmark, backtesting, metrics
    └─ head-to-head comparison of all 4 model variants
```

Each layer only depends on layers below it. Nothing in `models/` or `features/` knows about the CLI.

---

## Setup

```bash
cd 2026-world
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### Phase 1 — Predict and Simulate

**Predict a match (Elo + Poisson):**

```bash
python main.py predict Brazil France
python main.py predict Argentina England
```

**Simulate the tournament (default: top 16 teams):**

```bash
python main.py simulate
```

**Simulate with a custom team list:**

```bash
python main.py simulate Brazil France Argentina England Spain Portugal
```

### Phase 2 — Train and Evaluate

**Train the XGBoost ensemble model:**

```bash
python main.py train
```

Loads match data from `data/processed/matches.csv` (or falls back to `data/sample/matches.csv`), trains the model, and saves artifacts to `outputs/models/`. Prints a metrics summary table.

**Evaluate the trained model:**

```bash
python main.py evaluate
```

Loads the trained model, benchmarks all 4 model variants (Elo, Poisson, XGBoost, Ensemble) on the same held-out test set, and saves results to `outputs/evaluation/`.

**Detailed benchmark (shows winning model):**

```bash
python main.py benchmark
```

Same as `evaluate` but highlights the best model by log loss.

**Run tests:**

```bash
pytest tests/ -v
pytest --cov=src tests/ --cov-report=term-missing
```

---

## Sample Output

**Predict (Phase 1 — Elo + Poisson):**

```
┌─────────────────────────────┐
│   Brazil vs France          │
├──────────────────┬──────────┤
│ Brazil Win       │   42.3%  │
│ Draw             │   27.1%  │
│ France Win       │   30.6%  │
├──────────────────┼──────────┤
│ Expected Home xG │    1.72  │
│ Expected Away xG │    1.38  │
└──────────────────┴──────────┘
```

**Predict (Phase 2 — Ensemble, if model is trained):**

```
┌─────────────────────────────────────────────────────────────┐
│   Phase 2 Ensemble — Brazil vs France                       │
├───────────────┬──────────┬──────────┬──────────┬──────────┤
│ Outcome       │ Ensemble │ Elo      │ Poisson  │ XGBoost  │
├───────────────┼──────────┼──────────┼──────────┼──────────┤
│ Brazil Win    │   40.5%  │   42.3%  │   41.1%  │   38.2%  │
│ Draw          │   27.8%  │   27.1%  │   28.4%  │   27.6%  │
│ France Win    │   31.7%  │   30.6%  │   30.5%  │   34.2%  │
├───────────────┼──────────┼──────────┼──────────┼──────────┤
│ Confidence    │   82.4%  │          │          │          │
└───────────────┴──────────┴──────────┴──────────┴──────────┘
```

**Train output:**

```
┌───────────────────────┐
│   Training Results    │
├──────────────┬────────┤
│ Train size   │  1247  │
│ Test size    │   187  │
│ Accuracy     │ 0.5347 │
│ Log Loss     │ 0.9821 │
│ Brier Score  │ 0.5912 │
└──────────────┴────────┘
```

**Benchmark output:**

```
┌───────────────────────────────────────────────────────┐
│          Detailed Model Benchmark                     │
├──────────────────────┬──────────┬──────────┬─────────┤
│ Model                │ Accuracy │ Log Loss │ Brier   │
├──────────────────────┼──────────┼──────────┼─────────┤
│ Elo                  │ 0.5021   │ 1.0234   │ 0.6145  │
│ Poisson              │ 0.5134   │ 1.0081   │ 0.6023  │
│ XGBoost              │ 0.5347   │ 0.9821   │ 0.5912  │
│ Ensemble <-- BEST    │ 0.5289   │ 0.9756   │ 0.5878  │
└──────────────────────┴──────────┴──────────┴─────────┘
```

---

## Evaluation Metrics

| Metric | Description | Better when |
|--------|-------------|-------------|
| **Accuracy** | % of matches where predicted winner matches actual | Higher |
| **Log Loss** | Penalises confident wrong predictions; probabilistic accuracy | Lower |
| **Brier Score** | Mean squared error between predicted probs and one-hot true outcomes | Lower |

Ensemble and XGBoost typically beat Elo/Poisson on log loss and Brier score because they incorporate form, head-to-head, and other contextual features.

---

## Training Instructions

1. Ensure you have match data in `data/processed/matches.csv` (or the sample at `data/sample/matches.csv` will be used)
2. Run `python main.py train` — artifacts saved to `outputs/models/`
3. Run `python main.py evaluate` to see benchmark results
4. After training, `python main.py predict <home> <away>` will automatically show both Phase 1 and Phase 2 predictions

---

## Roadmap

| Phase | Focus |
|-------|-------|
| Phase 1 ✅ | Elo + Poisson + Monte Carlo CLI |
| Phase 2 ✅ | XGBoost ensemble, feature engineering, evaluation/backtesting |
| Phase 3 | Telegram bot + Google Sheets live dashboard |
| Phase 4 | News intelligence agent (LLM + web scraping) |
| Phase 5 | TimesFM time-series forecasting integration |

---

## Design Principles

- No hardcoded values — all parameters in `src/config/settings.py`
- All models independently testable
- Outputs always written to `outputs/` (never in-place)
- Logging via loguru to `outputs/logs/`
- Chronological train/test splits — no data leakage
