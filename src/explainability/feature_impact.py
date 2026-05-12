from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from src.explainability.shap_engine import SHAPEngine


class FeatureImpactRanker:
    """Rank features by global SHAP importance and optionally format as markdown."""

    def __init__(self, shap_engine: SHAPEngine) -> None:
        self._engine = shap_engine

    def rank(self, X: pd.DataFrame, top_n: int = 10) -> list[dict]:
        """Return top_n features sorted by mean |SHAP|.

        Returns list of dicts: [{"rank": 1, "feature": "elo_diff", "importance": 0.12}, ...]
        """
        global_imp = self._engine.global_shap(X)
        sorted_features = sorted(global_imp.items(), key=lambda x: x[1], reverse=True)
        return [
            {"rank": i + 1, "feature": name, "importance": round(val, 6)}
            for i, (name, val) in enumerate(sorted_features[:top_n])
        ]

    def to_markdown(self, ranked: list[dict]) -> str:
        lines = ["## Global Feature Importance (SHAP)\n"]
        lines.append("| Rank | Feature | Mean |SHAP| |")
        lines.append("|------|---------|--------------|")
        for entry in ranked:
            lines.append(f"| {entry['rank']} | {entry['feature']} | {entry['importance']:.6f} |")
        return "\n".join(lines)
