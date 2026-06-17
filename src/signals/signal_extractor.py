from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.signals.article_parser import ParsedArticle


@dataclass
class RawSignal:
    article_title: str
    article_link: str
    source_name: str
    signal_type: str
    matched_phrase: str
    context_snippet: str
    raw_text: str


# Ordered from most specific to least specific within each type
_PATTERNS: dict[str, list[str]] = {
    "injury": [
        r"\bruled out\b",
        r"\binjur(?:y|ied|ies)\b",
        r"\bdoubt(?:ful)?\b",
        r"\bfitness concern\b",
        r"\bmuscle strain\b",
        r"\bhamstring\b",
        r"\bknee(?:\s+injury)?\b",
        r"\blimp(?:ing)?\b",
    ],
    "suspension": [
        r"\bsuspend(?:ed|sion)\b",
        r"\bban(?:ned)?\b",
        r"\bred card\b",
        r"\baccumulation of yellow cards\b",
        r"\bunavailable\b",
    ],
    "lineup_change": [
        r"\blineup\b",
        r"\bstarting (?:xi|eleven|lineup)\b",
        r"\bbench(?:ed)?\b",
        r"\brotation\b",
        r"\bsquad change\b",
    ],
    "form": [
        r"\bwin streak\b",
        r"\bunbeaten\b",
        r"\bloss streak\b",
        r"\bpoor form\b",
        r"\bgood form\b",
        r"\bmomentum\b",
        r"\bconfidence\b",
    ],
    "weather": [
        r"\bheat\b",
        r"\bhumidity\b",
        r"\brain\b",
        r"\bwind\b",
        r"\bconditions\b",
    ],
    "travel": [
        r"\btravel(?:ling|ing)?\b",
        r"\blong(?:\s+haul)? flight\b",
        r"\bjet lag\b",
        r"\bfatigue\b",
    ],
}


def _extract_context(text: str, match: re.Match, window: int = 80) -> str:
    start = max(0, match.start() - window)
    end = min(len(text), match.end() + window)
    snippet = text[start:end].replace("\n", " ").strip()
    return f"…{snippet}…"


def extract_signals(article: ParsedArticle) -> list[RawSignal]:
    text = article.full_text
    signals: list[RawSignal] = []
    seen: set[str] = set()

    for sig_type, patterns in _PATTERNS.items():
        for pattern in patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                key = f"{sig_type}:{m.group(0).lower()}"
                if key in seen:
                    continue
                seen.add(key)
                signals.append(RawSignal(
                    article_title=article.title,
                    article_link=article.link,
                    source_name=article.source_name,
                    signal_type=sig_type,
                    matched_phrase=m.group(0),
                    context_snippet=_extract_context(text, m),
                    raw_text=text,
                ))

    return signals


def extract_all(articles: list[ParsedArticle]) -> list[RawSignal]:
    all_signals: list[RawSignal] = []
    for article in articles:
        all_signals.extend(extract_signals(article))
    return all_signals
