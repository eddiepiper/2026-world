from dataclasses import dataclass, field

import pandas as pd

from src.config.settings import EloConfig
from src.utils.helpers import normalize_probabilities


@dataclass
class EloModel:
    config: EloConfig = field(default_factory=EloConfig)
    ratings: dict[str, float] = field(default_factory=dict)

    def get_team_rating(self, team: str) -> float:
        return self.ratings.get(team, self.config.initial_rating)

    def expected_result(self, home_team: str, away_team: str) -> float:
        """Expected score for home team (0–1), including home advantage."""
        home_r = self.get_team_rating(home_team) + self.config.home_advantage
        away_r = self.get_team_rating(away_team)
        return 1.0 / (1.0 + 10.0 ** ((away_r - home_r) / 400.0))

    def update_ratings(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
    ) -> None:
        """Update both teams' ratings based on match result."""
        e_home = self.expected_result(home_team, away_team)

        if home_goals > away_goals:
            actual = 1.0
        elif home_goals == away_goals:
            actual = 0.5
        else:
            actual = 0.0

        home_r = self.get_team_rating(home_team)
        away_r = self.get_team_rating(away_team)

        self.ratings[home_team] = home_r + self.config.k_factor * (actual - e_home)
        self.ratings[away_team] = away_r + self.config.k_factor * ((1 - actual) - (1 - e_home))

    def win_draw_loss_probabilities(
        self,
        home_team: str,
        away_team: str,
        draw_rate: float = 0.25,
    ) -> tuple[float, float, float]:
        """Return (p_home_win, p_draw, p_away_win) using a fixed draw rate."""
        e = self.expected_result(home_team, away_team)
        raw = [e * (1 - draw_rate), draw_rate, (1 - e) * (1 - draw_rate)]
        win, draw, loss = normalize_probabilities(raw)
        return win, draw, loss

    def train_on_matches(self, matches_df: pd.DataFrame) -> None:
        """Replay historical matches chronologically to build ratings."""
        for _, row in matches_df.sort_values("date").iterrows():
            self.update_ratings(
                row["home_team"],
                row["away_team"],
                int(row["home_goals"]),
                int(row["away_goals"]),
            )
