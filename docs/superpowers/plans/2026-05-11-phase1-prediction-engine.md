# World Cup 2026 AI Prediction Engine — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular, production-grade World Cup 2026 AI prediction engine with data ingestion, Elo ratings, Poisson xG modeling, match prediction, and Monte Carlo tournament simulation.

**Architecture:** Modular Python package under `src/` with clean separation: `ingestion` → `models` → `simulation` → `CLI`. Each layer is independently testable and coupled only through typed interfaces.

**Tech Stack:** Python 3.11+, pandas, numpy, scipy, rich, loguru, pytest

---

## File Map

| File | Responsibility |
|------|---------------|
| `src/config/settings.py` | Centralized config dataclasses (Elo, Poisson, Simulation) |
| `src/utils/logger.py` | Loguru setup; writes to `outputs/logs/` |
| `src/utils/helpers.py` | `safe_div`, `normalize_probabilities`, `clamp` |
| `src/utils/metrics.py` | Brier score, prediction accuracy |
| `src/ingestion/match_loader.py` | Load + validate match CSV |
| `src/ingestion/team_loader.py` | Load + validate team metadata CSV |
| `src/ingestion/fifa_loader.py` | Load + validate FIFA ranking CSV |
| `src/models/elo_model.py` | Elo engine: expected_result, update_ratings, win_draw_loss |
| `src/models/poisson_model.py` | Poisson xG: fit, predict_goals, win_draw_loss_from_poisson |
| `src/models/prediction_engine.py` | Blend Elo + Poisson with configurable weights |
| `src/simulation/monte_carlo.py` | Monte Carlo bracket: run 10k simulations, output winner/finalist/semi probs |
| `src/main.py` | Core CLI logic: `predict` and `simulate` commands |
| `main.py` | Root entry point (imports src.main) |
| `data/sample/*.csv` | Realistic sample data (WC 2022 matches) |
| `tests/test_elo.py` | Elo model unit tests |
| `tests/test_poisson.py` | Poisson model unit tests |
| `tests/test_simulation.py` | Monte Carlo simulation unit tests |

---

### Task 1: Project Config + Utilities

**Files:**
- Create: `src/config/settings.py`
- Create: `src/utils/logger.py`
- Create: `src/utils/helpers.py`
- Create: `src/utils/metrics.py`
- Create: All `__init__.py` files

- [ ] Write `src/config/settings.py` with EloConfig, PoissonConfig, SimulationConfig, Settings dataclasses
- [ ] Write `src/utils/helpers.py` with safe_div, normalize_probabilities, clamp
- [ ] Write `src/utils/logger.py` with setup_logging using loguru
- [ ] Write `src/utils/metrics.py` with brier_score and accuracy
- [ ] Create all `__init__.py` files
- [ ] Run: `python -c "from src.config.settings import settings; print(settings)"` — expect dataclass output

---

### Task 2: Sample Data

**Files:**
- Create: `data/sample/matches.csv` — WC 2022 group+knockout matches
- Create: `data/sample/teams.csv` — 32 teams with confederation
- Create: `data/sample/fifa_rankings.csv` — current top 32 rankings

- [ ] Write matches.csv with 64 WC 2022 matches (date, home_team, away_team, home_goals, away_goals, tournament)
- [ ] Write teams.csv (team_name, confederation, country_code)
- [ ] Write fifa_rankings.csv (rank, team, points, date)
- [ ] Verify: `wc -l data/sample/matches.csv` — expect 65+ lines

---

### Task 3: Data Ingestion

**Files:**
- Create: `src/ingestion/match_loader.py`
- Create: `src/ingestion/team_loader.py`
- Create: `src/ingestion/fifa_loader.py`

- [ ] Write match_loader.py: load_matches() with null/duplicate/negative-score validation
- [ ] Write team_loader.py: load_teams() with column validation
- [ ] Write fifa_loader.py: load_fifa_rankings() with rank validation
- [ ] Run: `python -c "from src.ingestion.match_loader import load_matches; from pathlib import Path; df = load_matches(Path('data/sample/matches.csv')); print(len(df), 'matches')"` — expect 64 matches

---

### Task 4: Elo Model + Tests

**Files:**
- Create: `src/models/elo_model.py`
- Create: `tests/test_elo.py`

- [ ] Write EloModel dataclass: get_team_rating, expected_result, update_ratings, win_draw_loss_probabilities, train_on_matches
- [ ] Write tests: initial rating, expected result range, rating update direction, win/draw/loss sums to 1
- [ ] Run: `pytest tests/test_elo.py -v` — all pass

---

### Task 5: Poisson Model + Tests

**Files:**
- Create: `src/models/poisson_model.py`
- Create: `tests/test_poisson.py`

- [ ] Write PoissonModel: fit(), predict_goals(), simulate_score(), win_draw_loss_from_poisson()
- [ ] Write tests: avg_goals set after fit, strengths positive, probabilities sum to 1, unknown team uses defaults
- [ ] Run: `pytest tests/test_poisson.py -v` — all pass

---

### Task 6: Prediction Engine

**Files:**
- Create: `src/models/prediction_engine.py`

- [ ] Write PredictionResult TypedDict
- [ ] Write PredictionEngine: blends Elo + Poisson with configurable elo_weight, outputs normalized probabilities
- [ ] Run: `python -c "..."` smoke test — expect JSON-like output

---

### Task 7: Monte Carlo Simulation + Tests

**Files:**
- Create: `src/simulation/monte_carlo.py`
- Create: `tests/test_simulation.py`

- [ ] Write MonteCarloSimulator: simulate_knockout_match, simulate_bracket, run()
- [ ] Track winner/finalist/semifinalist per simulation
- [ ] Write tests: winner probabilities sum to 1, finalist ≥ winner, semifinal ≥ finalist
- [ ] Run: `pytest tests/test_simulation.py -v` — all pass

---

### Task 8: CLI + Documentation

**Files:**
- Create: `src/main.py`
- Create: `main.py` (root entry point)
- Create: `README.md`
- Create: `CLAUDE.md`
- Create: `requirements.txt`, `.env.example`, `.gitignore`

- [ ] Write CLI with predict and simulate commands using rich formatting
- [ ] Write README with setup, commands, roadmap
- [ ] Write CLAUDE.md with conventions and architecture
- [ ] Run: `python main.py predict Brazil France` — expect rich table output
- [ ] Run: `python main.py simulate` — expect simulation table
- [ ] Run: `pytest --cov=src tests/ -v` — expect ≥80% coverage
- [ ] `git init && git add . && git commit -m "feat: Phase 1 complete"`
