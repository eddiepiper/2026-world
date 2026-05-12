from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class UpsetFlag:
    home_team: str
    away_team: str
    is_upset: bool
    reasons: list[str] = field(default_factory=list)


class UpsetAnalyzer:
    """Flag matches with meaningful upset potential or high forecast uncertainty.

    A match is flagged (is_upset=True) if ANY of the following apply:
    - The underdog's win probability exceeds underdog_threshold
    - The confidence band is "Low"
    - The stability band is "Unstable"
    """

    def __init__(self, underdog_threshold: float = 0.35) -> None:
        self._threshold = underdog_threshold

    def check(
        self,
        home_team: str,
        away_team: str,
        home_win_prob: float,
        away_win_prob: float,
        confidence_band: str,
        stability_band: str,
    ) -> UpsetFlag:
        reasons: list[str] = []

        if home_win_prob < away_win_prob:
            underdog, underdog_prob = home_team, home_win_prob
        else:
            underdog, underdog_prob = away_team, away_win_prob

        if underdog_prob > self._threshold:
            reasons.append(
                f"{underdog} win probability {underdog_prob:.1%} "
                f"exceeds underdog threshold {self._threshold:.0%}"
            )
        if confidence_band == "Low":
            reasons.append("Low confidence band — prediction reliability is poor")
        if stability_band == "Unstable":
            reasons.append("Unstable prediction — high sensitivity to input variation")

        return UpsetFlag(
            home_team=home_team,
            away_team=away_team,
            is_upset=len(reasons) > 0,
            reasons=reasons,
        )
