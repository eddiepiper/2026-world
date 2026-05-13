from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from src.signals.signal_extractor import RawSignal

# Severity ordering
SEVERITY_RANK: dict[str, int] = {
    "critical": 3,
    "significant": 2,
    "minor": 1,
}

# Type → default severity when no strength modifier is present
_BASE_SEVERITY: dict[str, str] = {
    "injury": "significant",
    "suspension": "critical",
    "lineup_change": "minor",
    "form": "minor",
    "weather": "minor",
    "travel": "minor",
}

# Phrases that elevate severity one step
_ELEVATING_PHRASES = {"ruled out", "banned", "red card", "key player", "captain", "star"}
# Phrases that reduce severity one step
_REDUCING_PHRASES = {"doubt", "minor", "slight", "precaution", "managed"}


def _adjust_severity(base: str, context: str) -> str:
    lower = context.lower()
    rank = SEVERITY_RANK[base]
    if any(p in lower for p in _ELEVATING_PHRASES):
        rank = min(rank + 1, 3)
    if any(p in lower for p in _REDUCING_PHRASES):
        rank = max(rank - 1, 1)
    return {v: k for k, v in SEVERITY_RANK.items()}[rank]


@dataclass
class ClassifiedSignal:
    signal_type: str
    severity: str           # critical | significant | minor
    matched_phrase: str
    context_snippet: str
    article_title: str
    article_link: str
    source_name: str
    team_hint: str          # raw team name from context (before team_matcher resolves it)
    classified_at: str

    def to_dict(self) -> dict:
        return {
            "signal_type": self.signal_type,
            "severity": self.severity,
            "matched_phrase": self.matched_phrase,
            "context_snippet": self.context_snippet,
            "article_title": self.article_title,
            "article_link": self.article_link,
            "source_name": self.source_name,
            "team_hint": self.team_hint,
            "classified_at": self.classified_at,
        }


def _extract_team_hint(context: str, article_title: str) -> str:
    """Best-effort team hint from headline (resolved properly by team_matcher)."""
    return article_title.split(":")[0].strip()[:60]


def classify(raw: RawSignal) -> ClassifiedSignal:
    base = _BASE_SEVERITY.get(raw.signal_type, "minor")
    severity = _adjust_severity(base, raw.context_snippet)
    return ClassifiedSignal(
        signal_type=raw.signal_type,
        severity=severity,
        matched_phrase=raw.matched_phrase,
        context_snippet=raw.context_snippet,
        article_title=raw.article_title,
        article_link=raw.article_link,
        source_name=raw.source_name,
        team_hint=_extract_team_hint(raw.context_snippet, raw.article_title),
        classified_at=datetime.now(timezone.utc).isoformat(),
    )


def classify_all(raw_signals: list[RawSignal]) -> list[ClassifiedSignal]:
    return [classify(s) for s in raw_signals]


def save_classified(
    signals: list[ClassifiedSignal],
    processed_dir: Path,
    date_str: str | None = None,
) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = processed_dir / f"classified_{date_str}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in signals], f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(signals)} classified signals to {out_path}")
    return out_path


def load_classified(processed_dir: Path, date_str: str | None = None) -> list[ClassifiedSignal]:
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = processed_dir / f"classified_{date_str}.json"
    if not path.exists():
        logger.warning(f"No classified signals file for {date_str}")
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        ClassifiedSignal(**{k: v for k, v in d.items()})
        for d in data
    ]
