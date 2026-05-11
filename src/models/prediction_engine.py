from dataclasses import dataclass
from typing import TypedDict

from src.models.elo_model import EloModel
from src.models.poisson_model import PoissonModel
from src.utils.helpers import normalize_probabilities


class PredictionResult(TypedDict):
    home_team: str
    away_team: str
    win_probability: float
    draw_probability: float
    loss_probability: float
    expected_home_goals: float
    expected_away_goals: float


@dataclass
class PredictionEngine:
    elo_model: EloModel
    poisson_model: PoissonModel
    elo_weight: float = 0.5

    @property
    def poisson_weight(self) -> float:
        return 1.0 - self.elo_weight

    def predict(self, home_team: str, away_team: str) -> PredictionResult:
        elo_win, elo_draw, elo_loss = self.elo_model.win_draw_loss_probabilities(
            home_team, away_team
        )
        poi_win, poi_draw, poi_loss = self.poisson_model.win_draw_loss_from_poisson(
            home_team, away_team
        )

        raw_win = self.elo_weight * elo_win + self.poisson_weight * poi_win
        raw_draw = self.elo_weight * elo_draw + self.poisson_weight * poi_draw
        raw_loss = self.elo_weight * elo_loss + self.poisson_weight * poi_loss

        win, draw, loss = normalize_probabilities([raw_win, raw_draw, raw_loss])
        home_xg, away_xg = self.poisson_model.predict_goals(home_team, away_team)

        return PredictionResult(
            home_team=home_team,
            away_team=away_team,
            win_probability=round(win, 3),
            draw_probability=round(draw, 3),
            loss_probability=round(loss, 3),
            expected_home_goals=round(home_xg, 2),
            expected_away_goals=round(away_xg, 2),
        )
