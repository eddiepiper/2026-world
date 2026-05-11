from dataclasses import dataclass, field

import pandas as pd
from scipy.stats import poisson

from src.config.settings import PoissonConfig
from src.utils.helpers import safe_div


@dataclass
class PoissonModel:
    config: PoissonConfig = field(default_factory=PoissonConfig)
    attack_strength: dict[str, float] = field(default_factory=dict)
    defense_strength: dict[str, float] = field(default_factory=dict)
    avg_goals: float = 1.5

    def fit(self, matches_df: pd.DataFrame) -> None:
        """Calculate per-team attack/defense strengths from historical data."""
        total_goals = matches_df["home_goals"].sum() + matches_df["away_goals"].sum()
        n_matches = len(matches_df)
        self.avg_goals = safe_div(total_goals, 2 * n_matches, self.config.avg_goals_fallback)

        teams = set(matches_df["home_team"].unique()) | set(matches_df["away_team"].unique())
        for team in teams:
            home_m = matches_df[matches_df["home_team"] == team]
            away_m = matches_df[matches_df["away_team"] == team]

            scored = int(home_m["home_goals"].sum()) + int(away_m["away_goals"].sum())
            conceded = int(home_m["away_goals"].sum()) + int(away_m["home_goals"].sum())
            n = len(home_m) + len(away_m)

            avg_scored = safe_div(scored, n, self.avg_goals)
            avg_conceded = safe_div(conceded, n, self.avg_goals)

            self.attack_strength[team] = safe_div(avg_scored, self.avg_goals)
            self.defense_strength[team] = safe_div(avg_conceded, self.avg_goals)

    def predict_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Return (expected_home_goals, expected_away_goals)."""
        home_att = self.attack_strength.get(home_team, self.config.default_attack)
        away_def = self.defense_strength.get(away_team, self.config.default_defense)
        away_att = self.attack_strength.get(away_team, self.config.default_attack)
        home_def = self.defense_strength.get(home_team, self.config.default_defense)

        home_xg = self.avg_goals * home_att * away_def * self.config.home_advantage
        away_xg = self.avg_goals * away_att * home_def

        return home_xg, away_xg

    def simulate_score(self, home_team: str, away_team: str) -> tuple[int, int]:
        """Draw a single scoreline from the Poisson distribution."""
        home_xg, away_xg = self.predict_goals(home_team, away_team)
        return int(poisson.rvs(home_xg)), int(poisson.rvs(away_xg))

    def win_draw_loss_from_poisson(
        self,
        home_team: str,
        away_team: str,
        max_goals: int = 10,
    ) -> tuple[float, float, float]:
        """Compute win/draw/loss probs by summing scoreline probabilities."""
        home_xg, away_xg = self.predict_goals(home_team, away_team)
        p_win = p_draw = p_loss = 0.0

        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                p = float(poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg))
                if h > a:
                    p_win += p
                elif h == a:
                    p_draw += p
                else:
                    p_loss += p

        return p_win, p_draw, p_loss
