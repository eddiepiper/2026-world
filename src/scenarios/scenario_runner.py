from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from src.features.feature_builder import FeatureBuilder, FEATURE_COLUMNS
from src.ml.xgboost_model import XGBoostMatchModel
from src.models.elo_model import EloModel
from src.models.poisson_model import PoissonModel

_VALID_PARAMS = frozenset(
    {"attack_strength", "defense_strength", "elo_rating", "recent_form"}
)


@dataclass
class Perturbation:
    param: str
    team: str
    delta: float

    def validate(self) -> None:
        if self.param not in _VALID_PARAMS:
            raise ValueError(
                f"Unknown param '{self.param}'. "
                f"Valid options: {sorted(_VALID_PARAMS)}"
            )


class ScenarioRunner:
    def __init__(
        self,
        elo_model: EloModel,
        poisson_model: PoissonModel,
        xgb_model: XGBoostMatchModel,
        feature_builder: FeatureBuilder,
        weights: tuple[float, float, float] = (0.3, 0.3, 0.4),
    ) -> None:
        self._elo = elo_model
        self._poisson = poisson_model
        self._xgb = xgb_model
        self._feature_builder = feature_builder
        self._weights = weights

    def run(
        self,
        home_team: str,
        away_team: str,
        perturbations: list[Perturbation],
        match_date: pd.Timestamp | None = None,
    ) -> dict:
        """Return baseline vs perturbed ensemble probability delta table."""
        for p in perturbations:
            p.validate()

        if match_date is None:
            match_date = pd.Timestamp.today()

        baseline = self._blend(home_team, away_team, match_date)

        # Save state before perturbations
        # (only save for teams that appear in perturbations)
        perturb_teams = {p.team for p in perturbations}
        saved_elo = {}
        for team in perturb_teams:
            try:
                saved_elo[team] = self._elo.ratings.get(team, 1500.0)
            except AttributeError:
                saved_elo[team] = self._elo.get_team_rating(team)

        saved_attack = dict(getattr(self._poisson, 'attack_strength', {}))
        saved_defense = dict(getattr(self._poisson, 'defense_strength', {}))

        # Apply perturbations
        for p in perturbations:
            if p.param == "elo_rating":
                current = saved_elo.get(p.team, 1500.0)
                self._elo.ratings[p.team] = current + p.delta
            elif p.param == "attack_strength":
                current = getattr(self._poisson, 'attack_strength', {}).get(p.team, 1.0)
                self._poisson.attack_strength[p.team] = current + p.delta
            elif p.param == "defense_strength":
                current = getattr(self._poisson, 'defense_strength', {}).get(p.team, 1.0)
                self._poisson.defense_strength[p.team] = current + p.delta

        form_deltas = {
            p.team: p.delta for p in perturbations if p.param == "recent_form"
        }
        try:
            perturbed = self._blend(home_team, away_team, match_date, form_deltas=form_deltas)
        finally:
            # Restore
            for team, rating in saved_elo.items():
                self._elo.ratings[team] = rating
            if saved_attack:
                self._poisson.attack_strength = saved_attack
            if saved_defense:
                self._poisson.defense_strength = saved_defense

        delta = {
            k: round(perturbed[k] - baseline[k], 4)
            for k in ("home_win", "draw", "away_win")
        }

        logger.info(
            "ScenarioRunner.run: {} vs {} — {} perturbation(s) applied",
            home_team,
            away_team,
            len(perturbations),
        )

        return {
            "home_team": home_team,
            "away_team": away_team,
            "baseline": {k: round(v, 4) for k, v in baseline.items()},
            "perturbed": {k: round(v, 4) for k, v in perturbed.items()},
            "delta": delta,
            "perturbations": [
                {"param": p.param, "team": p.team, "delta": p.delta}
                for p in perturbations
            ],
        }

    def to_markdown(self, result: dict) -> str:
        home, away = result["home_team"], result["away_team"]
        lines = [
            f"## Scenario: {home} vs {away}\n",
            "| Outcome | Baseline | Perturbed | Delta |",
            "|---------|----------|-----------|-------|",
        ]
        for outcome in ("home_win", "draw", "away_win"):
            label = outcome.replace("_", " ").title()
            b = result["baseline"][outcome]
            p = result["perturbed"][outcome]
            d = result["delta"][outcome]
            sign = "+" if d >= 0 else ""
            lines.append(f"| {label} | {b:.3f} | {p:.3f} | {sign}{d:.3f} |")
        if result["perturbations"]:
            lines.append("\n**Perturbations applied:**")
            for pt in result["perturbations"]:
                sign = "+" if pt["delta"] >= 0 else ""
                lines.append(f"- {pt['team']} {pt['param']}: {sign}{pt['delta']}")
        return "\n".join(lines)

    def _blend(
        self,
        home_team: str,
        away_team: str,
        match_date: pd.Timestamp,
        form_deltas: dict[str, float] | None = None,
    ) -> dict[str, float]:
        ew, pw, xw = self._weights

        elo_w, elo_d, elo_l = self._elo.win_draw_loss_probabilities(home_team, away_team)
        poi_w, poi_d, poi_l = self._poisson.win_draw_loss_from_poisson(home_team, away_team)

        features = self._feature_builder.build_features_for_match(
            home_team, away_team, match_date
        )

        if form_deltas:
            if home_team in form_deltas:
                features["home_win_rate"] = (
                    features.get("home_win_rate", 0.0) + form_deltas[home_team]
                )
            if away_team in form_deltas:
                features["away_win_rate"] = (
                    features.get("away_win_rate", 0.0) + form_deltas[away_team]
                )

        feat_df = pd.DataFrame([features])
        xgb_result = self._xgb.predict_proba_dict(feat_df)[0]

        raw = [
            ew * elo_w + pw * poi_w + xw * xgb_result["home_win"],
            ew * elo_d + pw * poi_d + xw * xgb_result["draw"],
            ew * elo_l + pw * poi_l + xw * xgb_result["away_win"],
        ]
        total = sum(raw)
        if total > 0:
            raw = [v / total for v in raw]
        return {"home_win": raw[0], "draw": raw[1], "away_win": raw[2]}
