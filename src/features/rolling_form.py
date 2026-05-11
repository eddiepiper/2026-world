"""Rolling form statistics for each team, computed without data leakage.

All queries are strictly ``date < match_date`` so features for match M
only use data from matches played *before* M's date.
"""
from __future__ import annotations

import pandas as pd
from loguru import logger


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

        self._df = matches_df.copy()
        # Ensure date column is datetime
        self._df["date"] = pd.to_datetime(self._df["date"])
        self.window = window
        logger.debug(
            "RollingFormCalculator initialised with {} matches, window={}",
            len(self._df),
            window,
        )

    # ------------------------------------------------------------------
    # Core retrieval helper
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
        if n is None:
            n = self.window

        before_date = pd.Timestamp(before_date)

        home_mask = (self._df["home_team"] == team) & (self._df["date"] < before_date)
        away_mask = (self._df["away_team"] == team) & (self._df["date"] < before_date)

        home_rows = self._df[home_mask][["date", "home_goals", "away_goals"]].rename(
            columns={"home_goals": "goals_scored", "away_goals": "goals_conceded"}
        )
        away_rows = self._df[away_mask][["date", "home_goals", "away_goals"]].rename(
            columns={"away_goals": "goals_scored", "home_goals": "goals_conceded"}
        )

        combined = (
            pd.concat([home_rows, away_rows], ignore_index=True)
            .sort_values("date", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )
        return combined

    # ------------------------------------------------------------------
    # Public statistics API
    # ------------------------------------------------------------------

    def win_rate(self, team: str, before_date: pd.Timestamp) -> float:
        """Fraction of last ``self.window`` matches that were wins.

        Returns 0.0 when no matches exist before *before_date*.
        """
        recent = self.get_team_recent_matches(team, before_date)
        if recent.empty:
            return 0.0
        wins = (recent["goals_scored"] > recent["goals_conceded"]).sum()
        return float(wins / len(recent))

    def goals_scored_avg(self, team: str, before_date: pd.Timestamp) -> float:
        """Average goals scored per match in last ``self.window`` matches.

        Returns 0.0 when no matches exist before *before_date*.
        """
        recent = self.get_team_recent_matches(team, before_date)
        if recent.empty:
            return 0.0
        return float(recent["goals_scored"].mean())

    def goals_conceded_avg(self, team: str, before_date: pd.Timestamp) -> float:
        """Average goals conceded per match in last ``self.window`` matches.

        Returns 0.0 when no matches exist before *before_date*.
        """
        recent = self.get_team_recent_matches(team, before_date)
        if recent.empty:
            return 0.0
        return float(recent["goals_conceded"].mean())

    def clean_sheet_rate(self, team: str, before_date: pd.Timestamp) -> float:
        """Fraction of last ``self.window`` matches where team conceded 0 goals.

        Returns 0.0 when no matches exist before *before_date*.
        """
        recent = self.get_team_recent_matches(team, before_date)
        if recent.empty:
            return 0.0
        clean = (recent["goals_conceded"] == 0).sum()
        return float(clean / len(recent))

    def days_since_last_match(self, team: str, before_date: pd.Timestamp) -> float:
        """Days elapsed since team's most recent match before *before_date*.

        Returns 30.0 when no prior matches are found.
        """
        before_date = pd.Timestamp(before_date)
        recent = self.get_team_recent_matches(team, before_date, n=1)
        if recent.empty:
            return 30.0
        last_date = pd.Timestamp(recent.iloc[0]["date"])
        return float((before_date - last_date).days)
