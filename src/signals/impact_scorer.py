from __future__ import annotations

from dataclasses import dataclass

from src.signals.signal_classifier import ClassifiedSignal, SEVERITY_RANK

# Impact tiers (spec):
#   low      → 1–2 pp
#   medium   → 3–5 pp
#   high     → 6–8 pp
#   hard cap → 8 pp per signal, no exceptions
#
# These are review-layer deltas ONLY — they never touch the core model.
HARD_CAP: float = 0.08  # absolute ceiling per signal on any single outcome

_MAX_DELTA: dict[str, dict[str, float]] = {
    "injury": {
        "critical": 0.08,   # high
        "significant": 0.04,  # medium
        "minor": 0.01,        # low
    },
    "suspension": {
        "critical": 0.08,   # high (capped from 0.09 to respect hard cap)
        "significant": 0.05,  # medium
        "minor": 0.02,        # low
    },
    "lineup_change": {
        "critical": 0.05,   # medium
        "significant": 0.03,  # medium
        "minor": 0.01,        # low
    },
    "form": {
        "critical": 0.06,   # high
        "significant": 0.03,  # medium
        "minor": 0.01,        # low
    },
    "weather": {
        "critical": 0.02,   # low
        "significant": 0.01,  # low
        "minor": 0.01,        # low
    },
    "travel": {
        "critical": 0.02,   # low
        "significant": 0.01,  # low
        "minor": 0.01,        # low
    },
}

# Signals that affect home vs away team
# "negative" signals (injury/suspension) hurt the afflicted team
_NEGATIVE_TYPES = {"injury", "suspension"}
_NEUTRAL_TYPES = {"weather", "travel"}   # affect both teams equally → cancel out
_FORM_TYPES = {"form", "lineup_change"}  # direction depends on positive/negative framing


def _is_positive_form(context: str) -> bool:
    pos = {"win streak", "unbeaten", "good form", "momentum", "confidence", "dominant"}
    return any(p in context.lower() for p in pos)


@dataclass
class ImpactDelta:
    home_win_delta: float   # positive = helps home team
    draw_delta: float
    away_win_delta: float
    signal_type: str
    severity: str
    affected_team: str
    rationale: str

    def clamp(self, lo: float = -HARD_CAP, hi: float = HARD_CAP) -> "ImpactDelta":
        """Enforce per-signal hard cap of ±8 pp on any single outcome."""
        self.home_win_delta = max(lo, min(hi, self.home_win_delta))
        self.draw_delta = max(lo, min(hi, self.draw_delta))
        self.away_win_delta = max(lo, min(hi, self.away_win_delta))
        return self


def score_signal(
    signal: ClassifiedSignal,
    affected_team: str,
    home_team: str,
    away_team: str,
) -> ImpactDelta:
    """
    Compute review-layer probability deltas for a single signal.
    These deltas are display-only and must never mutate core model output.
    """
    sig_type = signal.signal_type
    severity = signal.severity
    max_d = _MAX_DELTA.get(sig_type, {}).get(severity, 0.01)
    is_home = affected_team.lower() == home_team.lower()

    home_d = draw_d = away_d = 0.0

    if sig_type in _NEGATIVE_TYPES:
        # Negative for the affected team → opponent benefits
        shift = max_d
        if is_home:
            home_d = -shift
            draw_d = shift * 0.3
            away_d = shift * 0.7
        else:
            away_d = -shift
            draw_d = shift * 0.3
            home_d = shift * 0.7

        rationale = (
            f"{signal.severity.title()} {sig_type} affecting {affected_team} "
            f"('{signal.matched_phrase}')"
        )

    elif sig_type in _FORM_TYPES:
        positive = _is_positive_form(signal.context_snippet)
        shift = max_d if positive else -max_d
        if is_home:
            home_d = shift
            draw_d = -shift * 0.4
            away_d = -shift * 0.6
        else:
            away_d = shift
            draw_d = -shift * 0.4
            home_d = -shift * 0.6
        direction = "positive" if positive else "negative"
        rationale = (
            f"{direction.title()} {sig_type} signal for {affected_team} "
            f"('{signal.matched_phrase}')"
        )

    else:
        # Neutral (weather, travel) — apply symmetrically; effect cancels
        rationale = f"Neutral {sig_type} signal ('{signal.matched_phrase}') — negligible net impact"

    return ImpactDelta(
        home_win_delta=home_d,
        draw_delta=draw_d,
        away_win_delta=away_d,
        signal_type=sig_type,
        severity=severity,
        affected_team=affected_team,
        rationale=rationale,
    ).clamp()


def aggregate_deltas(deltas: list[ImpactDelta]) -> tuple[float, float, float]:
    """Sum all impact deltas and return (home_d, draw_d, away_d), clamped per component."""
    home_d = sum(d.home_win_delta for d in deltas)
    draw_d = sum(d.draw_delta for d in deltas)
    away_d = sum(d.away_win_delta for d in deltas)
    # Global clamp: max ±0.20 total shift per outcome
    clamp = 0.20
    return (
        max(-clamp, min(clamp, home_d)),
        max(-clamp, min(clamp, draw_d)),
        max(-clamp, min(clamp, away_d)),
    )
