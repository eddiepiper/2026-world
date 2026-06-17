from __future__ import annotations

import re
from dataclasses import dataclass

from loguru import logger

# Canonical team name list (World Cup 2026 participants + common aliases)
KNOWN_TEAMS: list[str] = [
    "Argentina", "France", "Brazil", "England", "Spain", "Germany",
    "Netherlands", "Portugal", "Belgium", "Croatia", "Italy", "Uruguay",
    "Mexico", "United States", "Canada", "Ecuador", "Senegal", "Morocco",
    "Japan", "South Korea", "Australia", "Ghana", "Cameroon", "Nigeria",
    "Saudi Arabia", "Iran", "Qatar", "Tunisia", "Denmark", "Switzerland",
    "Poland", "Serbia", "Wales", "Costa Rica", "South Africa", "Panama",
    "Honduras", "Jamaica", "El Salvador", "Bolivia",
]

_ALIASES: dict[str, str] = {
    "usa": "United States",
    "u.s.": "United States",
    "u.s.a.": "United States",
    "us": "United States",
    "korea": "South Korea",
    "republic of korea": "South Korea",
    "dpr korea": "North Korea",
    "england": "England",
    "tres leones": "Mexico",
    "die mannschaft": "Germany",
    "la albiceleste": "Argentina",
    "les bleus": "France",
    "seleção": "Brazil",
    "the socceroos": "Australia",
    "the lions": "Senegal",
    "ivory coast": "Ivory Coast",
    "côte d'ivoire": "Ivory Coast",
}


@dataclass
class TeamMatch:
    canonical_name: str
    matched_text: str
    confidence: float


def _build_pattern() -> re.Pattern:
    names = sorted(KNOWN_TEAMS, key=len, reverse=True)  # longest first
    escaped = [re.escape(n) for n in names]
    alias_escaped = [re.escape(a) for a in sorted(_ALIASES.keys(), key=len, reverse=True)]
    all_terms = escaped + alias_escaped
    return re.compile(r"\b(" + "|".join(all_terms) + r")\b", re.IGNORECASE)


_TEAM_PATTERN = _build_pattern()


def find_teams_in_text(text: str) -> list[TeamMatch]:
    matches: list[TeamMatch] = []
    seen: set[str] = set()
    for m in _TEAM_PATTERN.finditer(text):
        raw = m.group(0)
        canonical = _resolve(raw)
        if canonical and canonical not in seen:
            seen.add(canonical)
            # Exact canonical match → 1.0; alias → 0.85
            conf = 1.0 if raw.title() in KNOWN_TEAMS else 0.85
            matches.append(TeamMatch(
                canonical_name=canonical,
                matched_text=raw,
                confidence=conf,
            ))
    return matches


def _resolve(raw: str) -> str | None:
    lower = raw.lower()
    if lower in _ALIASES:
        return _ALIASES[lower]
    title = raw.title()
    if title in KNOWN_TEAMS:
        return title
    # Case-insensitive fallback
    for team in KNOWN_TEAMS:
        if team.lower() == lower:
            return team
    return None


def match_signal_to_teams(text: str) -> list[str]:
    """Return canonical team names found in text."""
    return [m.canonical_name for m in find_teams_in_text(text)]
