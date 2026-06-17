"""
Generate a sample daily signal review report for Mexico vs South Africa.
Demonstrates the full signal pipeline with synthetic signals.
Run: python scripts/generate_demo_report.py
"""
from __future__ import annotations

from pathlib import Path

from src.signals.signal_classifier import ClassifiedSignal
from src.signals.signal_reporter import save_report
from src.signals.signal_review import BaseForecast, build_review

DEMO_DATE = "2026-05-13"
OUTPUT_DIR = Path("outputs/signals/reports")

DEMO_SIGNALS = [
    ClassifiedSignal(
        signal_type="injury",
        severity="significant",
        matched_phrase="injury",
        context_snippet=(
            "Mexico key striker Raul Jimenez has been ruled out with a knee injury "
            "and is a major doubt for the opening group stage match."
        ),
        article_title="Mexico striker Raul Jimenez ruled out with injury",
        article_link="https://example.com/mexico-injury",
        source_name="BBC Sport Football",
        team_hint="Mexico",
        classified_at=f"{DEMO_DATE}T08:00:00+00:00",
    ),
    ClassifiedSignal(
        signal_type="form",
        severity="minor",
        matched_phrase="win streak",
        context_snippet=(
            "South Africa are on a five-game win streak heading into the tournament, "
            "with confidence high in the squad after their AFCON qualifier campaign."
        ),
        article_title="South Africa riding high on confidence ahead of World Cup opener",
        article_link="https://example.com/south-africa-form",
        source_name="Goal.com",
        team_hint="South Africa",
        classified_at=f"{DEMO_DATE}T09:30:00+00:00",
    ),
]

if __name__ == "__main__":
    base = BaseForecast(
        home_team="Mexico",
        away_team="South Africa",
        home_win_prob=0.56,
        draw_prob=0.25,
        away_win_prob=0.19,
    )

    review = build_review(base, DEMO_SIGNALS)

    print(f"\nBase forecast:      Mexico {base.home_win_prob:.0%} | Draw {base.draw_prob:.0%} | SA {base.away_win_prob:.0%}")
    print(f"Signal-adjusted:    Mexico {review.adjusted_home_win:.0%} | Draw {review.adjusted_draw:.0%} | SA {review.adjusted_away_win:.0%}")
    print(f"\nSignals applied: {len(review.signals_applied)}")
    print(f"\n{review.disclaimer}\n")

    path = save_report(review, OUTPUT_DIR, date_str=DEMO_DATE)
    print(f"Report saved to: {path}")
