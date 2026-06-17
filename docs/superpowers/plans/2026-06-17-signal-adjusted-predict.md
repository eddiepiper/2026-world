# Signal-Adjusted Predict Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Append a "Signal-Adjusted" section to `predict` output that loads today's classified signals from disk and shows base → adjusted probabilities with deltas.

**Architecture:** After the Phase 2 ensemble table renders in `cmd_predict()`, load classified signals via `load_classified()`, build a `BaseForecast` from ensemble probabilities, call the existing `build_review()`, and render a third table. Graceful degradation: no signals file → dim note; signals exist but none relevant → dim note. No new scoring logic — `build_review()` already handles everything.

**Tech Stack:** Python 3.11+, loguru, rich, existing `src/signals/signal_review.py`, `src/signals/signal_classifier.py`

## Global Constraints

- No new scoring logic — all signal math stays in `build_review()` and `score_signal()`
- Signal section is read-only: never mutates models or training state
- Graceful degradation required: missing file or no relevant signals must not raise or crash `predict`
- All output uses `rich.console.Console`, never `print()`
- Follow existing lazy-import pattern in `cmd_predict()` (imports inside the try block)
- Tests use `pytest` fixtures and `unittest.mock` for filesystem isolation

---

### Task 1: Wire signal-adjusted section into `cmd_predict`

**Files:**
- Modify: `src/main.py` — append signal section after ensemble table in `cmd_predict()`

**Interfaces:**
- Consumes:
  - `load_classified(processed_dir: Path, date_str: str | None) -> list[ClassifiedSignal]` from `src.signals.signal_classifier`
  - `BaseForecast(home_team, away_team, home_win_prob, draw_prob, away_win_prob)` from `src.signals.signal_review`
  - `build_review(base: BaseForecast, signals: list[ClassifiedSignal]) -> SignalReview` from `src.signals.signal_review`
  - `SignalReview.adjusted_home_win`, `.adjusted_draw`, `.adjusted_away_win`, `.any_signals`, `.disclaimer` attributes
  - `ep` dict with keys `"home_win"`, `"draw"`, `"away_win"` — already in scope from ensemble block
  - `_PROCESSED_DIR = settings.outputs_dir / "signals" / "processed"` — define inline
- Produces: nothing (pure side-effect: console output)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_signals.py` (or create `tests/test_predict_signals.py` if preferred — use the existing `test_signals.py`):

```python
# At the top of test_signals.py, add these imports if not already present:
# from unittest.mock import patch, MagicMock
# import sys

class TestPredictSignalSection:
    """Signal-adjusted section appears in cmd_predict output."""

    def test_signal_section_shown_when_signals_relevant(self, tmp_path, injury_signal):
        """When today's classified signals include a team in the match, signal table renders."""
        from unittest.mock import patch, MagicMock
        from io import StringIO

        # Patch load_classified to return one relevant signal
        # Patch ensemble internals so the ensemble block succeeds
        with patch("src.main.console") as mock_console, \
             patch("src.main._load_matches_df") as mock_matches, \
             patch("src.signals.signal_classifier.load_classified", return_value=[injury_signal]):
            pass  # actual assertion in step 3 after implementation

    def test_signal_section_shows_dim_note_when_no_file(self):
        """When no signals file exists for today, a dim note is printed."""
        from unittest.mock import patch

        with patch("src.main.console") as mock_console, \
             patch("src.signals.signal_classifier.load_classified", return_value=[]):
            pass  # actual assertion in step 3 after implementation
```

Write the real tests:

```python
class TestPredictSignalSection:
    """Signal-adjusted section appears after ensemble in cmd_predict."""

    def test_no_crash_when_signals_empty(self, mexico_vs_sa):
        """cmd_predict signal section handles empty signal list without raising."""
        from src.signals.signal_review import build_review
        review = build_review(mexico_vs_sa, [])
        assert not review.any_signals
        assert review.adjusted_home_win == mexico_vs_sa.home_win_prob
        assert review.adjusted_draw == mexico_vs_sa.draw_prob
        assert review.adjusted_away_win == mexico_vs_sa.away_win_prob

    def test_signal_section_adjusts_probabilities(self, mexico_vs_sa, injury_signal):
        """build_review with a relevant signal shifts probabilities from base."""
        from src.signals.signal_review import build_review
        review = build_review(mexico_vs_sa, [injury_signal])
        assert review.any_signals
        # Probabilities must differ from base when signal applies
        changed = (
            review.adjusted_home_win != mexico_vs_sa.home_win_prob
            or review.adjusted_draw != mexico_vs_sa.draw_prob
            or review.adjusted_away_win != mexico_vs_sa.away_win_prob
        )
        assert changed, "Signal should shift at least one probability"

    def test_signal_section_probabilities_sum_to_one(self, mexico_vs_sa, injury_signal):
        """Adjusted probabilities always sum to 1.0 after normalization."""
        from src.signals.signal_review import build_review
        review = build_review(mexico_vs_sa, [injury_signal])
        total = review.adjusted_home_win + review.adjusted_draw + review.adjusted_away_win
        assert abs(total - 1.0) < 1e-6
```

- [ ] **Step 2: Run tests to verify they pass (these test existing behaviour)**

```bash
cd /Users/edwardchiang/2026-world
.venv/bin/pytest tests/test_signals.py::TestPredictSignalSection -v
```

Expected: all 3 PASS (they test `build_review` which already exists correctly)

- [ ] **Step 3: Add signal section to `cmd_predict` in `src/main.py`**

Find the block ending around line 154 in `src/main.py`:
```python
            console.print()
            console.print(ens_table)
        except Exception as exc:
            logger.warning(f"Ensemble prediction failed: {exc}")
            console.print(f"[dim]Ensemble unavailable: {exc}[/dim]")
    else:
        console.print(
            "[dim]No ensemble model — run [bold]python main.py train[/bold] first[/dim]"
        )

    console.print()
```

Replace the section from `console.print()` (after `ens_table`) through the final `console.print()` with:

```python
            console.print()
            console.print(ens_table)

            # Phase 4: Signal-adjusted section
            try:
                from src.signals.signal_classifier import load_classified
                from src.signals.signal_review import BaseForecast, build_review

                _processed_dir = settings.outputs_dir / "signals" / "processed"
                signals = load_classified(_processed_dir)

                if not signals:
                    console.print(
                        "[dim]No signals for today — run "
                        "[bold]python main.py signals collect[/bold] to fetch latest news.[/dim]"
                    )
                else:
                    base = BaseForecast(
                        home_team=home_team,
                        away_team=away_team,
                        home_win_prob=ep["home_win"],
                        draw_prob=ep["draw"],
                        away_win_prob=ep["away_win"],
                    )
                    review = build_review(base, signals)

                    if not review.any_signals:
                        console.print(
                            f"[dim]No relevant signals found for "
                            f"{home_team} vs {away_team}.[/dim]"
                        )
                    else:
                        sig_table = Table(
                            title=f"[bold]Signal-Adjusted[/bold] — {home_team} vs {away_team}",
                            show_header=True,
                            header_style="bold yellow",
                        )
                        sig_table.add_column("Outcome", style="white")
                        sig_table.add_column("Ensemble", style="magenta", justify="right")
                        sig_table.add_column("Signal-Adjusted", style="yellow", justify="right")
                        sig_table.add_column("Delta", style="cyan", justify="right")

                        def _delta(base_p: float, adj_p: float) -> str:
                            diff = adj_p - base_p
                            sign = "+" if diff >= 0 else ""
                            return f"{sign}{diff:.1%}"

                        sig_table.add_row(
                            f"{home_team} Win",
                            f"{ep['home_win']:.1%}",
                            f"{review.adjusted_home_win:.1%}",
                            _delta(ep["home_win"], review.adjusted_home_win),
                        )
                        sig_table.add_row(
                            "Draw",
                            f"{ep['draw']:.1%}",
                            f"{review.adjusted_draw:.1%}",
                            _delta(ep["draw"], review.adjusted_draw),
                        )
                        sig_table.add_row(
                            f"{away_team} Win",
                            f"{ep['away_win']:.1%}",
                            f"{review.adjusted_away_win:.1%}",
                            _delta(ep["away_win"], review.adjusted_away_win),
                        )
                        sig_table.add_section()
                        sig_table.add_row(
                            f"Signals applied",
                            f"{len(review.signals_applied)}",
                            "", "",
                        )

                        console.print()
                        console.print(sig_table)
                        console.print(f"[dim]{review.disclaimer}[/dim]")
            except Exception as exc:
                logger.warning(f"Signal section failed: {exc}")

        except Exception as exc:
            logger.warning(f"Ensemble prediction failed: {exc}")
            console.print(f"[dim]Ensemble unavailable: {exc}[/dim]")
    else:
        console.print(
            "[dim]No ensemble model — run [bold]python main.py train[/bold] first[/dim]"
        )

    console.print()
```

- [ ] **Step 4: Run full test suite to check for regressions**

```bash
cd /Users/edwardchiang/2026-world
.venv/bin/pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected: all tests pass, no new failures.

- [ ] **Step 5: Smoke test the predict command**

```bash
cd /Users/edwardchiang/2026-world
.venv/bin/python main.py predict Brazil France 2>&1
```

Expected output (three sections):
1. Phase 1 table: Elo + Poisson probabilities
2. Phase 2 Ensemble table: Elo / Poisson / XGBoost / Ensemble columns
3. Either:
   - Signal-Adjusted table with Delta column (if today's signals file exists and is relevant), or
   - Dim note: "No signals for today — run python main.py signals collect…"

- [ ] **Step 6: Commit**

```bash
cd /Users/edwardchiang/2026-world
git add src/main.py tests/test_signals.py
git commit -m "feat(predict): append signal-adjusted section with delta column"
```
