# Phase 4 — Signal Intelligence Layer

> Branch: `experiment/worldcup-news-signals`
> Status: Experimental — read-only review layer, not merged to main

---

## Purpose

The Signal Intelligence Layer provides **world-state awareness** for the forecasting engine. It collects external football signals (injuries, suspensions, form, etc.) from RSS feeds, classifies them, maps them to teams, and generates a **daily signal-adjusted review** note.

Signals surface analyst-facing context. They never mutate core model probabilities, ensemble weights, trained artifacts, or benchmark outputs.

---

## Architecture

```
src/signals/
├── source_config.py      RSS source registry (name, URL, tags, enabled flag)
├── rss_collector.py      Feed fetcher → outputs/signals/raw/signals_YYYY-MM-DD.json
├── article_parser.py     HTML stripping, relevance keyword filtering
├── signal_extractor.py   Regex-based signal detection (6 signal types)
├── signal_classifier.py  Severity scoring: critical / significant / minor
├── team_matcher.py       Canonical team name resolution + alias map
├── impact_scorer.py      Bounded probability delta engine
├── signal_review.py      Builds signal-adjusted review (immutable base)
└── signal_reporter.py    Markdown + JSON output renderer
```

```
src/cli/signal_commands.py   CLI subcommands wired into main.py
outputs/signals/
├── raw/          Raw RSS JSON (one file per day)
├── processed/    Classified signal JSON (one file per day)
└── reports/      Signal review markdown reports
```

---

## Signal Types

| Type | Examples | Default severity |
|------|----------|-----------------|
| `injury` | ruled out, knee injury, hamstring | significant |
| `suspension` | banned, red card, 5-yellow accumulation | critical |
| `lineup_change` | rotation, benched, squad change | minor |
| `form` | win streak, unbeaten, poor form | minor |
| `weather` | heat, humidity, rain | minor |
| `travel` | long-haul flight, jet lag, fatigue | minor |

---

## Impact Scoring Caps

All deltas are **display-only** and never touch the core model.

| Tier | Range |
|------|-------|
| Low | 1–2 pp |
| Medium | 3–5 pp |
| High | 6–8 pp |
| **Hard cap** | **8 pp per signal, per outcome** |
| Aggregate cap | 20 pp total shift across all signals |

---

## CLI Reference

```bash
# Fetch RSS feeds → outputs/signals/raw/
python main.py signals collect

# Classify raw signals → outputs/signals/processed/
python main.py signals classify

# Build signal review for a match
python main.py signals review Mexico "South Africa"

# Generate daily signal summary JSON
python main.py signals report

# Full daily pipeline (collect → classify → review → report)
python main.py signals run_daily Mexico "South Africa"
```

---

## Demo: Mexico vs South Africa

**Base forecast (core model — unchanged):**

| Outcome | Probability |
|---------|------------|
| Mexico Win | 56% |
| Draw | 25% |
| South Africa Win | 19% |

**Signal: Mexico key striker injury (significant)**

**Signal-adjusted review:**

| Outcome | Base | Adjusted | Delta |
|---------|------|----------|-------|
| Mexico Win | 56% | 51% | −5% |
| Draw | 25% | 26% | +1% |
| South Africa Win | 19% | 23% | +4% |

> ⚠️ **Signal-adjusted review only. Core model probability unchanged.**

---

## Isolation Guarantee

- Signals write only to `outputs/signals/`
- `build_review()` is a pure in-memory function — no filesystem side effects
- `BaseForecast` is never mutated
- No signal code imports from or writes to `outputs/models/`, `outputs/evaluation/`, or `outputs/simulations/`
- Tests enforce all of the above (see `tests/test_signals.py`)

---

## Adding a New Signal Source

Edit `src/signals/source_config.py`:

```python
SignalSource(
    name="My Source",
    url="https://example.com/feed.xml",
    category="rss",
    tags=["injuries", "world cup"],
    enabled=True,
)
```

Requires `feedparser`: `pip install feedparser`

---

## Test Coverage

38+ tests in `tests/test_signals.py` covering:
- Team matching (canonical names, aliases, case-insensitive)
- Signal classification (all 6 types × 3 severities)
- Hard cap enforcement (no signal > 8 pp)
- Probability normalization (always sums to 1.0)
- No-mutation guarantees (base forecast, core output dirs)
- Graceful degradation (missing feeds, missing files, feedparser absent)
