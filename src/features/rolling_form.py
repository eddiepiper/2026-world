"""Rolling form statistics for each team, computed without data leakage.

All queries are strictly ``date < match_date`` so features for match M
only use data from matches played *before* M's date.
"""
from __future__ import annotations

from bisect import bisect_left

import pandas as pd
from loguru import logger

from src.utils.helpers import safe_div


class RollingFormCalculator:
    """Compute per-team rolling statistics using a fixed look-back window."""

    def __init__(self, matches_df: pd.DataFrame, window: int = 5) -> None:
        """
        Parameters
        ----------
        matches_df:
            Full historical match DataFrame with columns:
            date, home_team, away_team, home_goals, away_goals.
        window:
            Number of recent matches to look back (default 5).
        """
        required = {"date", "home_team", "away_team", "home_goals", "away_goals"}
        missing = required - set(matches_df.columns)
        if missing:
            raise ValueError(f"matches_df missing columns: {missing}")

        self._df = matches_df.sort_values("date").reset_index(drop=True)
        self._df["date"] = pd.to_datetime(self._df["date"])
        self.window = window

        # Pre-index: team -> list of (date, goals_scored, goals_conceded) sorted by date
        self._team_index: dict[str, list[tuple[pd.Timestamp, int, int]]] = {}
        self._team_dates: dict[str, list[pd.Timestamp]] = {}
        self._build_index()

        logger.debug(
            "RollingFormCalculator initialised with {} matches, window={}",
            len(self._df),
            window,
        )

    def _build_index(self) -> None:
        """Build per-team sorted index from the match DataFrame (O(n))."""
        for _, row in self._df.iterrows():
            for team, scored, conceded in [
                (row["home_team"], row["home_goals"], row["away_goals"]),
                (row["away_team"], row["away_goals"], row["home_goals"]),
            ]:
                if team not in self._team_index:
                    self._team_index[team] = []
                    self._team_dates[team] = []
                self._team_index[team].append((row["date"], int(scored), int(conceded)))
                self._team_dates[team].append(row["date"])

    def _get_recent(
        self,
        team: str,
        before_date: pd.Timestamp,
        n: int | None = None,
    ) -> list[tuple[pd.Timestamp, int, int]]:
        """Return last *n* entries for *team* strictly before *before_date*.

        Uses binary search (O(log k)) into the pre-built sorted index.
        """
        window = n if n is not None else self.window
        entries = self._team_index.get(team, [])
        dates = self._team_dates.get(team, [])
        cutoff = bisect_left(dates, before_date)
        return entries[max(0, cutoff - window):cutoff]

    # ------------------------------------------------------------------
    # Core retrieval helper (public, for backward compatibility)
    # ------------------------------------------------------------------

    def get_team_recent_matches(
        self,
        team: str,
        before_date: pd.Timestamp,
        n: int | None = None,
    ) -> pd.DataFrame:
        """Return the last *n* matches for *team* strictly before *before_date*.

        Columns returned:
            date, goals_scored, goals_conceded

        Sorted most-recent-first.  If *n* is None, ``self.window`` is used.
        """
        before_date = pd.Timestamp(before_date)
        recent = self._get_recent(team, before_date, n)
        if not recent:
            return pd.DataFrame(columns=["date", "goals_scored", "goals_conceded"])

        rows = [
            {"date": d, "goals_scored": scored, "goals_conceded": conceded}
            for d, scored, conceded in reversed(recent)  # most-recent-first
        ]
        return pd.DataFrame(rows).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Public statistics API
    # ------------------------------------------------------------------

    def win_rate(self, team: str, before_date: pd.Timestamp) -> float:
        """Fraction of last ``self.window`` matches that were wins.

        Returns 0.0 when no matches exist before *before_date*.
        """
        recent = self._get_recent(team, pd.Timestamp(before_date))
        wins = sum(1 for _, scored, conceded in recent if scored > conceded)
        return safe_div(wins, len(recent), 0.0)

    def goals_scored_avg(self, team: str, before_date: pd.Timestamp) -> float:
        """Average goals scored per match in last ``self.window`` matches.

        Returns 0.0 when no matches exist before *before_date*.
        """
        recent = self._get_recent(team, pd.Timestamp(before_date))
        if not recent:
            return 0.0
        return float(sum(scored for _, scored, _ in recent) / len(recent))

    def goals_conceded_avg(self, team: str, before_date: pd.Timestamp) -> float:
        """Average goals conceded per match in last ``self.window`` matches.

        Returns 0.0 when no matches exist before *before_date*.
        """
        recent = self._get_recent(team, pd.Timestamp(before_date))
        if not recent:
            return 0.0
        return float(sum(conceded for _, _, conceded in recent) / len(recent))

    def clean_sheet_rate(self, team: str, before_date: pd.Timestamp) -> float:
        """Fraction of last ``self.window`` matches where team conceded 0 goals.

        Returns 0.0 when no matches exist before *before_date*.
        """
        recent = self._get_recent(team, pd.Timestamp(before_date))
        clean = sum(1 for _, _, conceded in recent if conceded == 0)
        return safe_div(clean, len(recent), 0.0)

    def days_since_last_match(self, team: str, before_date: pd.Timestamp) -> float:
        """Days elapsed since team's most recent match before *before_date*.

        Returns 30.0 when no prior matches are found.
        """
        before_date = pd.Timestamp(before_date)
        recent = self._get_recent(team, before_date, n=1)
        if not recent:
            return 30.0
        last_date = recent[-1][0]  # last entry is the most recent (before cutoff)
        return float((before_date - last_date).days)
