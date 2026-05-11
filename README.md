# 2026-world — World Cup AI Prediction Engine

A modular, production-grade AI system for football match prediction, tournament simulation, and football intelligence research. Built as an AI engineering portfolio project.

---

## What it does

- **Elo Rating Engine** — dynamic team ratings updated from historical match data
- **Poisson xG Model** — attack/defense strength + home advantage → expected goals
- **Match Prediction** — blended Elo + Poisson probabilities with configurable weights
- **Monte Carlo Simulation** — 10,000-run tournament bracket → winner/finalist/semifinal probabilities

---

## Architecture

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

Each layer only depends on layers below it. `prediction_engine` combines models; `monte_carlo` uses `prediction_engine`. Nothing in `models/` knows about the CLI.

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

**Predict a match:**

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

**Run tests:**

```bash
pytest tests/ -v
pytest --cov=src tests/ --cov-report=term-missing
```

---

## Sample Output

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

---

## Roadmap

| Phase | Focus |
|-------|-------|
| Phase 1 ✅ | Elo + Poisson + Monte Carlo CLI |
| Phase 2 | XGBoost ensemble, feature engineering, backtesting |
| Phase 3 | Telegram bot + Google Sheets live dashboard |
| Phase 4 | News intelligence agent (LLM + web scraping) |
| Phase 5 | TimesFM time-series forecasting integration |

---

## Design Principles

- No hardcoded values — all parameters in `src/config/settings.py`
- All models independently testable
- Outputs always written to `outputs/` (never in-place)
- Logging via loguru to `outputs/logs/`
