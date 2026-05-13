from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from src.signals.signal_classifier import ClassifiedSignal
from src.signals.signal_review import SignalReview

_SEVERITY_EMOJI: dict[str, str] = {
    "critical": "🔴",
    "significant": "🟡",
    "minor": "🟢",
}


def _fmt_prob(p: float) -> str:
    return f"{p:.0%}"


def _pct_delta(base: float, adjusted: float) -> str:
    diff = adjusted - base
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.0%}"


def render_signal_list(signals: list[ClassifiedSignal], top_n: int = 10) -> str:
    if not signals:
        return "_No relevant signals found for this match._\n"
    lines = []
    for s in signals[:top_n]:
        icon = _SEVERITY_EMOJI.get(s.severity, "⚪")
        lines.append(
            f"- {icon} **[{s.severity.upper()}]** `{s.signal_type}` — "
            f"_{s.matched_phrase}_ via *{s.source_name}*\n"
            f"  > {s.context_snippet[:120]}"
        )
    return "\n".join(lines) + "\n"


def render_review_table(review: SignalReview) -> str:
    b = review.base
    lines = [
        "| Outcome | Base Forecast | Signal-Adjusted Review | Delta |",
        "|---------|:-------------:|:----------------------:|:-----:|",
        f"| **{b.home_team} Win** | {_fmt_prob(b.home_win_prob)} "
        f"| {_fmt_prob(review.adjusted_home_win)} "
        f"| {_pct_delta(b.home_win_prob, review.adjusted_home_win)} |",
        f"| **Draw** | {_fmt_prob(b.draw_prob)} "
        f"| {_fmt_prob(review.adjusted_draw)} "
        f"| {_pct_delta(b.draw_prob, review.adjusted_draw)} |",
        f"| **{b.away_team} Win** | {_fmt_prob(b.away_win_prob)} "
        f"| {_fmt_prob(review.adjusted_away_win)} "
        f"| {_pct_delta(b.away_win_prob, review.adjusted_away_win)} |",
    ]
    return "\n".join(lines) + "\n"


def render_full_report(review: SignalReview, date_str: str | None = None) -> str:
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    b = review.base
    sections = [
        f"# Signal Intelligence Review — {b.home_team} vs {b.away_team}",
        f"_Generated: {date_str}_\n",
        "## Base Forecast (Core Model — Unchanged)",
        render_review_table(review),
        f"> ⚠️ **{review.disclaimer}**\n",
        f"## Signals Applied ({len(review.signals_applied)} relevant signals)",
        render_signal_list(review.signals_applied),
    ]

    if review.deltas:
        sections.append("## Net Impact Summary")
        net = review.net_delta_summary()
        sections.append(
            f"- Home win shift: {_pct_delta(0, net['home_win'])}\n"
            f"- Draw shift: {_pct_delta(0, net['draw'])}\n"
            f"- Away win shift: {_pct_delta(0, net['away_win'])}\n"
        )
        sections.append("### Signal Rationale")
        for d in review.deltas:
            sections.append(f"- {d.rationale}")
        sections.append("")

    sections.append("---")
    sections.append(
        "_This report is for analyst review only. "
        "Signal-adjusted probabilities do not feed into the tournament simulation, "
        "ensemble model, or any stored forecast outputs._"
    )

    return "\n\n".join(sections)


def save_report(
    review: SignalReview,
    reports_dir: Path,
    date_str: str | None = None,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    b = review.base
    safe_match = f"{b.home_team}_vs_{b.away_team}".replace(" ", "_")
    out_path = reports_dir / f"signal_review_{safe_match}_{date_str}.md"

    content = render_full_report(review, date_str)
    out_path.write_text(content, encoding="utf-8")
    logger.info(f"Signal review report saved to {out_path}")
    return out_path


def save_classified_json(
    signals: list[ClassifiedSignal],
    reports_dir: Path,
    date_str: str | None = None,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = reports_dir / f"signal_classified_{date_str}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in signals], f, indent=2, ensure_ascii=False)
    logger.info(f"Classified signals JSON saved to {out_path}")
    return out_path
