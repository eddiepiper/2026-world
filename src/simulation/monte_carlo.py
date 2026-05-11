from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.prediction_engine import PredictionEngine


@dataclass
class MonteCarloSimulator:
    engine: PredictionEngine
    n_simulations: int = 10_000
    random_seed: int | None = None

    def simulate_knockout_match(self, home_team: str, away_team: str) -> str:
        """Simulate a single knockout match. Drawn games go to penalties (50/50)."""
        result = self.engine.predict(home_team, away_team)
        r = np.random.random()
        if r < result["win_probability"]:
            return home_team
        elif r < result["win_probability"] + result["draw_probability"]:
            return str(np.random.choice([home_team, away_team]))
        return away_team

    def simulate_bracket(self, teams: list[str]) -> dict:
        """
        Simulate a single-elimination bracket.

        Tracks semifinalists (final 4) and finalists (final 2).
        Pads to the next power of 2 if the team count is not a power of 2.
        """
        if len(teams) < 2:
            raise ValueError("Need at least 2 teams to simulate a bracket.")

        # Pad to next power of 2
        target = 1
        while target < len(teams):
            target *= 2
        current = list(teams) + [teams[-1]] * (target - len(teams))

        semifinalists: list[str] = []
        finalists: list[str] = []

        while len(current) > 1:
            if len(current) == 4:
                semifinalists = list(current)
            if len(current) == 2:
                finalists = list(current)

            next_round: list[str] = []
            for i in range(0, len(current), 2):
                winner = self.simulate_knockout_match(current[i], current[i + 1])
                next_round.append(winner)
            current = next_round

        return {
            "winner": current[0],
            "finalists": finalists,
            "semifinalists": semifinalists,
        }

    def run(self, teams: list[str]) -> pd.DataFrame:
        """
        Run n_simulations and return a DataFrame with winner/finalist/semifinal
        probabilities for each team, sorted by winner probability descending.
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        win_counts: dict[str, int] = {t: 0 for t in teams}
        finalist_counts: dict[str, int] = {t: 0 for t in teams}
        semi_counts: dict[str, int] = {t: 0 for t in teams}

        for _ in range(self.n_simulations):
            result = self.simulate_bracket(teams)
            win_counts[result["winner"]] = win_counts.get(result["winner"], 0) + 1
            for t in result["finalists"]:
                finalist_counts[t] = finalist_counts.get(t, 0) + 1
            for t in result["semifinalists"]:
                semi_counts[t] = semi_counts.get(t, 0) + 1

        n = self.n_simulations
        rows = [
            {
                "team": team,
                "winner_prob": round(win_counts[team] / n, 4),
                "finalist_prob": round(finalist_counts[team] / n, 4),
                "semifinal_prob": round(semi_counts[team] / n, 4),
            }
            for team in teams
        ]

        return (
            pd.DataFrame(rows)
            .sort_values("winner_prob", ascending=False)
            .reset_index(drop=True)
        )
