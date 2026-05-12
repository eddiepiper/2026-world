from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from src.explainability.shap_engine import SHAPEngine


class PredictionExplainer:
    """Format local SHAP values into human-readable explanations.

    Saves both markdown and JSON to outputs/reports/explanations/.
    """

    def __init__(self, shap_engine: SHAPEngine) -> None:
        self._engine = shap_engine

    def explain(
        self,
        home_team: str,
        away_team: str,
        feature_row: pd.DataFrame,
        outcome: str = "H",
        top_n: int = 5,
    ) -> dict:
        """Return top-N SHAP drivers for the given match and outcome.

        Returns:
            {
                "home_team": "Brazil",
                "away_team": "France",
                "outcome_explained": "H",
                "top_drivers": [
                    {"rank": 1, "feature": "elo_diff", "shap_value": 0.18, "raw_value": 120.5},
                    ...
                ],
            }
        """
        local = self._engine.local_shap(feature_row, outcome=outcome)
        return {
            "home_team": home_team,
            "away_team": away_team,
            "outcome_explained": outcome,
            "top_drivers": [
                {"rank": i + 1, **entry}
                for i, entry in enumerate(local[:top_n])
            ],
        }

    def to_markdown(self, explanation: dict) -> str:
        home = explanation["home_team"]
        away = explanation["away_team"]
        outcome_label = {"H": "Home win", "D": "Draw", "A": "Away win"}.get(
            explanation["outcome_explained"], explanation["outcome_explained"]
        )
        lines = [f"## {home} vs {away} — Top Drivers ({outcome_label})\n"]
        for driver in explanation["top_drivers"]:
            sign = "+" if driver["shap_value"] >= 0 else ""
            lines.append(
                f"{driver['rank']}. {driver['feature']:<28} "
                f"{sign}{driver['shap_value']:.4f}"
            )
        return "\n".join(lines)

    def save(
        self,
        explanation: dict,
        output_dir: Path,
        match_date: str | None = None,
    ) -> None:
        """Save explanation as both .md and .json."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        home = explanation["home_team"].replace(" ", "_")
        away = explanation["away_team"].replace(" ", "_")
        date_str = match_date or pd.Timestamp.today().strftime("%Y-%m-%d")
        stem = f"{home}_vs_{away}_{date_str}"

        md_path = output_dir / f"{stem}.md"
        json_path = output_dir / f"{stem}.json"

        md_path.write_text(self.to_markdown(explanation))
        json_path.write_text(json.dumps(explanation, indent=2))

        logger.info(f"Explanation saved to {md_path} and {json_path}")
