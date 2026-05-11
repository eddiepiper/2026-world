# CLAUDE.md — 2026-world

AI coding assistant instructions for this project.

---

## Project Overview

`2026-world` is a modular World Cup AI prediction engine. Phase 1 is complete. Do not add Phase 2+ features unless the user explicitly asks.

---

## Architecture

| Layer | Package | Depends on |
|-------|---------|------------|
| Config | `src/config/` | nothing |
| Utils | `src/utils/` | config |
| Ingestion | `src/ingestion/` | utils |
| Models | `src/models/` | config, utils |
| Simulation | `src/simulation/` | models |
| CLI | `src/main.py` | everything above |

**Rule:** Never introduce circular dependencies. Lower layers must not import from higher layers.

---

## Module Responsibilities

- `settings.py` — single source of truth for all numeric parameters (K-factor, home advantage, weights, n_simulations). If you're hardcoding a number, it belongs here.
- `elo_model.py` — pure Elo logic. No I/O, no pandas dependency beyond `train_on_matches`.
- `poisson_model.py` — pure Poisson xG logic. `fit()` takes a DataFrame; everything else is numeric.
- `prediction_engine.py` — combines Elo + Poisson. Returns `PredictionResult` TypedDict. No I/O.
- `monte_carlo.py` — calls `prediction_engine.predict()` in a loop. Writes nothing; caller saves CSV.
- `match_loader.py` / `team_loader.py` / `fifa_loader.py` — load, validate, return clean DataFrames. Raise `ValueError` on structural issues, log warnings for data quality issues.
- `main.py` — CLI only. Wires up models, calls commands, formats output with `rich`. No business logic here.

---

## Coding Conventions

- Python 3.11+. Use `int | None` union syntax, not `Optional[int]`.
- Dataclasses (`@dataclass`) for models and config. No plain dicts for structured state.
- `TypedDict` for function return shapes that cross module boundaries.
- Type hints everywhere. No `Any` without a comment explaining why.
- `safe_div()` instead of bare division when denominator could be zero.
- `normalize_probabilities()` whenever blending raw probability components.
- All file I/O goes through `pathlib.Path`. No bare strings as paths.
- Log with `loguru.logger`, not `print`. `logger.info` for normal operations, `logger.warning` for data issues, `logger.error` for caught exceptions.

---

## Testing

- Tests live in `tests/`. One file per module: `test_elo.py`, `test_poisson.py`, `test_simulation.py`.
- Use `pytest.fixture` for shared setup (trained models, sample DataFrames).
- Mock `PredictionEngine` with `unittest.mock.MagicMock` in simulation tests — don't call real models.
- Target: ≥80% coverage on `src/`. Run: `pytest --cov=src tests/ --cov-report=term-missing`.
- Do not write tests that depend on randomness without setting `random_seed`.

---

## Anti-Patterns to Avoid

- **No hardcoded team names or ratings** in model code. Use `settings.py` or pass them as arguments.
- **No `print()`** — use `rich.console.Console` in CLI, `loguru.logger` in library code.
- **No fat main()** — keep `src/main.py` as a thin CLI wrapper. Extract helpers if logic grows.
- **No in-place data mutation** — loaders return clean copies; don't modify the caller's DataFrame.
- **No Phase 2+ code** in Phase 1 files. Don't add XGBoost, Telegram, or LLM imports.

---

## Future Phases (do not implement until asked)

- **Phase 2** — XGBoost ensemble, feature engineering, backtesting framework
- **Phase 3** — Telegram bot, Google Sheets live export
- **Phase 4** — News intelligence agent (LLM + scraping)
- **Phase 5** — TimesFM time-series forecasting

---

## Common Commands

```bash
# Run CLI
python main.py predict Brazil France
python main.py simulate

# Tests
pytest tests/ -v
pytest --cov=src tests/ --cov-report=term-missing

# Install deps
pip install -r requirements.txt
```
