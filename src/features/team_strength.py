"""Team strength features wrapping pre-trained Elo and Poisson models.

This module does NOT train models — it only reads from already-trained
``EloModel`` and ``PoissonModel`` instances passed to the constructor.
"""
from __future__ import annotations

import pandas as pd
from loguru import logger

from src.models.elo_model import EloModel
from src.models.poisson_model import PoissonModel


class TeamStrengthFeatures:
    """Provide numeric strength features for a pair of teams."""

    _DEFAULT_FIFA_POINTS: float = 1000.0
    _DEFAULT_ATTACK: float = 1.0
    _DEFAULT_DEFENSE: float = 1.0

    def __init__(
        self,
        elo_model: EloModel,
        poisson_model: PoissonModel | None = None,
        rankings_df: pd.DataFrame | None = None,
    ) -> None:
        """
        Parameters
        ----------
        elo_model:
            A trained ``EloModel`` instance.
        poisson_model:
            An optional trained ``PoissonModel`` instance.  When *None*,
            ``attack_strength`` and ``defense_strength`` always return 1.0.
        rankings_df:
            Optional DataFrame with columns ``[team, points]`` for FIFA ranking
            data.  When *None* or when a team is missing, ``fifa_points``
            returns the default of 1000.0.
        """
        self._elo = elo_model
        self._poisson = poisson_model

        # Build FIFA lookup dict
        self._fifa: dict[str, float] = {}
        if rankings_df is not None:
            if not {"team", "points"}.issubset(rankings_df.columns):
                logger.warning(
                    "rankings_df missing 'team' or 'points' column — FIFA features disabled"
                )
            else:
                self._fifa = dict(
                    zip(rankings_df["team"], rankings_df["points"])
                )
                logger.debug(
                    "Loaded FIFA rankings for {} teams", len(self._fifa)
                )

    # ------------------------------------------------------------------
    # Elo features
    # ------------------------------------------------------------------

    def elo_rating(self, team: str) -> float:
        """Return team's current Elo rating."""
        return self._elo.get_team_rating(team)

    def elo_diff(self, home_team: str, away_team: str) -> float:
        """Return home_elo − away_elo."""
        return self.elo_rating(home_team) - self.elo_rating(away_team)

    # ------------------------------------------------------------------
    # FIFA ranking features
    # ------------------------------------------------------------------

    def fifa_points(self, team: str) -> float:
        """Return FIFA ranking points.  Returns 1000.0 if team not in rankings."""
        return self._fifa.get(team, self._DEFAULT_FIFA_POINTS)

    def fifa_points_diff(self, home_team: str, away_team: str) -> float:
        """Return home_points − away_points."""
        return self.fifa_points(home_team) - self.fifa_points(away_team)

    # ------------------------------------------------------------------
    # Poisson features
    # ------------------------------------------------------------------

    def attack_strength(self, team: str) -> float:
        """Return team's Poisson attack strength.  Returns 1.0 if unknown."""
        if self._poisson is None:
            return self._DEFAULT_ATTACK
        return self._poisson.attack_strength.get(team, self._DEFAULT_ATTACK)

    def defense_strength(self, team: str) -> float:
        """Return team's Poisson defense strength.  Returns 1.0 if unknown."""
        if self._poisson is None:
            return self._DEFAULT_DEFENSE
        return self._poisson.defense_strength.get(team, self._DEFAULT_DEFENSE)
