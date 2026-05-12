# CLAUDE.md — 2026-world

AI coding assistant instructions for this project.

---

## Project Overview

`2026-world` is a modular World Cup AI prediction engine. Phase 1 (Elo + Poisson + Monte Carlo CLI) and Phase 2 (ML ensemble with XGBoost, feature engineering, calibration, evaluation) are both complete.

---

## Architecture

| Layer | Package | Depends on |
|-------|---------|------------|
| Config | `src/config/` | nothing |
| Utils | `src/utils/` | config |
| Ingestion | `src/ingestion/` | utils |
| Models | `src/models/` | config, utils |
| Features | `src/features/` | models, config |
| ML | `src/ml/` | features, models |
| Ensemble | `src/ensemble/` | ml, models |
| Evaluation | `src/evaluation/` | ml, models, features |
| Simulation | `src/simulation/` | models |
| CLI | `src/main.py` | everything above |

**Rule:** Never introduce circular dependencies. Lower layers must not import from higher layers.

---

## Module Responsibilities

### Phase 1 Modules

- `settings.py` — single source of truth for all numeric parameters (K-factor, home advantage, weights, n_simulations). If you're hardcoding a number, it belongs here.
- `elo_model.py` — pure Elo logic. No I/O, no pandas dependency beyond `train_on_matches`.
- `poisson_model.py` — pure Poisson xG logic. `fit()` takes a DataFrame; everything else is numeric.
- `prediction_engine.py` — combines Elo + Poisson. Returns `PredictionResult` TypedDict. No I/O.
- `monte_carlo.py` — calls `prediction_engine.predict()` in a loop. Writes nothing; caller saves CSV.
- `match_loader.py` / `team_loader.py` / `fifa_loader.py` — load, validate, return clean DataFrames. Raise `ValueError` on structural issues, log warnings for data quality issues.
- `main.py` — CLI only. Wires up models, calls commands, formats output with `rich`. No business logic here.

### Phase 2 Modules

- `feature_builder.py` — builds the feature matrix for ML training and inference. Computes Elo, Poisson, rolling form, and team strength features per match.
- `rolling_form.py` — computes rolling win/draw/loss rates and goal statistics per team.
- `team_strength.py` — computes attack/defense strength indices from historical match data.
- `xgboost_model.py` — multiclass XGBoost classifier (H/D/A). Wraps XGBClassifier with sklearn Pipeline and LabelEncoder. Saves/loads model artifacts to disk.
- `trainer.py` — orchestrates feature building, train/test split (chronological), fitting, evaluation, and artifact saving. `run()` returns metrics dict.
- `calibration.py` — wraps a trained model with sklearn's `CalibratedClassifierCV` to improve probability calibration.
- `ensemble_engine.py` — blends Elo + Poisson + XGBoost probabilities with configurable weights. Computes confidence score. Top-level prediction module for Phase 2.
- `benchmark.py` — compares all 4 models (Elo, Poisson, XGBoost, Ensemble) on the same chronological test set. Saves CSV + markdown.
- `backtesting.py` — walk-forward backtesting framework for evaluating prediction accuracy over time.
- `metrics.py` — shared metrics helpers (accuracy, log_loss, brier_score, compute_all_metrics).

---

## Coding Conventions

- Python 3.11+. Use `int | None` union syntax, not `Optional[int]`.
- Dataclasses (`@dataclass`) for models and config. No plain dicts for structured state.
- `TypedDict` for function return shapes that cross module boundaries.
- Type hints everywhere. No `Any` without a comment explaining why.
- `safe_div()` instead of bare division when denominator could be zero.
- `normalize_probabilities()` whenever blending raw probability components.
- All file I/O goes through `pathlib.Path`. No bare strings as paths.
- Log with `loguru.logger`, not `print`. `logger.info` for normal operations, `logger.warning` for data quality issues, `logger.error` for caught exceptions.

---

## Testing

- Tests live in `tests/`. One file per module: `test_elo.py`, `test_poisson.py`, `test_simulation.py`.
- Use `pytest.fixture` for shared setup (trained models, sample DataFrames).
- Mock `PredictionEngine` with `unittest.mock.MagicMock` in simulation tests — don't call real models.
- Target: 80%+ coverage on `src/`. Run: `pytest --cov=src tests/ --cov-report=term-missing`.
- Do not write tests that depend on randomness without setting `random_seed`.

---

## Anti-Patterns to Avoid

- **No hardcoded team names or ratings** in model code. Use `settings.py` or pass them as arguments.
- **No `print()`** — use `rich.console.Console` in CLI, `loguru.logger` in library code.
- **No fat main()** — keep `src/main.py` as a thin CLI wrapper. Extract helpers if logic grows.
- **No in-place data mutation** — loaders return clean copies; don't modify the caller's DataFrame.

---

## Future Phases (do not implement until asked)

- **Phase 3** — Telegram bot, Google Sheets live export
- **Phase 4** — News intelligence agent (LLM + scraping)
- **Phase 5** — TimesFM time-series forecasting

---

## Common Commands

```bash
# Run CLI
python main.py predict Brazil France
python main.py simulate
python main.py train
python main.py evaluate
python main.py benchmark

# Tests
pytest tests/ -v
pytest --cov=src tests/ --cov-report=term-missing

# Install deps
pip install -r requirements.txt
```
