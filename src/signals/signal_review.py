from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from src.signals.signal_classifier import ClassifiedSignal
from src.signals.impact_scorer import ImpactDelta, aggregate_deltas, score_signal
from src.signals.team_matcher import match_signal_to_teams

REVIEW_DISCLAIMER = (
    "Signal-adjusted review only. Core model probability unchanged."
)


@dataclass
class BaseForecast:
    home_team: str
    away_team: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float


@dataclass
class SignalReview:
    base: BaseForecast
    signals_applied: list[ClassifiedSignal]
    deltas: list[ImpactDelta]
    adjusted_home_win: float
    adjusted_draw: float
    adjusted_away_win: float
    disclaimer: str = REVIEW_DISCLAIMER

    @property
    def any_signals(self) -> bool:
        return len(self.signals_applied) > 0

    def net_delta_summary(self) -> dict[str, float]:
        h, d, a = aggregate_deltas(self.deltas)
        return {"home_win": h, "draw": d, "away_win": a}


def _normalize(h: float, d: float, a: float) -> tuple[float, float, float]:
    total = h + d + a
    if total <= 0:
        return 1 / 3, 1 / 3, 1 / 3
    return h / total, d / total, a / total


def build_review(
    base: BaseForecast,
    signals: list[ClassifiedSignal],
) -> SignalReview:
    """
    Apply signal deltas to base forecast probabilities for review display.
    Never mutates the base forecast or any model state.
    """
    relevant: list[ClassifiedSignal] = []
    deltas: list[ImpactDelta] = []

    for sig in signals:
        # Resolve which teams are mentioned in the signal context
        teams_in_signal = match_signal_to_teams(
            sig.context_snippet + " " + sig.article_title
        )
        # Keep only signals that mention at least one of the match teams
        match_teams = {base.home_team.lower(), base.away_team.lower()}
        affected = [t for t in teams_in_signal if t.lower() in match_teams]
        if not affected:
            continue

        sig_deltas: list[ImpactDelta] = []
        for team in affected:
            delta = score_signal(sig, team, base.home_team, base.away_team)
            sig_deltas.append(delta)

        # Only count this signal if it produced at least one non-zero delta
        has_impact = any(
            abs(d.home_win_delta) > 0 or abs(d.draw_delta) > 0 or abs(d.away_win_delta) > 0
            for d in sig_deltas
        )
        if has_impact:
            deltas.extend(sig_deltas)
            relevant.append(sig)

    h_adj = base.home_win_prob
    d_adj = base.draw_prob
    a_adj = base.away_win_prob

    if deltas:
        net_h, net_d, net_a = aggregate_deltas(deltas)
        h_adj = base.home_win_prob + net_h
        d_adj = base.draw_prob + net_d
        a_adj = base.away_win_prob + net_a
        # Clamp to [0, 1] before normalizing
        h_adj = max(0.0, h_adj)
        d_adj = max(0.0, d_adj)
        a_adj = max(0.0, a_adj)
        h_adj, d_adj, a_adj = _normalize(h_adj, d_adj, a_adj)

    logger.info(
        f"Signal review for {base.home_team} vs {base.away_team}: "
        f"{len(relevant)} relevant signals applied"
    )

    return SignalReview(
        base=base,
        signals_applied=relevant,
        deltas=deltas,
        adjusted_home_win=round(h_adj, 4),
        adjusted_draw=round(d_adj, 4),
        adjusted_away_win=round(a_adj, 4),
    )
