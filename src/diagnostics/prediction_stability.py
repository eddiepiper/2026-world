from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.config.settings import DiagnosticsConfig, settings
from src.features.feature_builder import FEATURE_COLUMNS, FeatureBuilder
from src.ml.xgboost_model import XGBoostMatchModel
from src.models.elo_model import EloModel
from src.models.poisson_model import PoissonModel


class StabilityAnalyzer:
    """Perturbation-based sensitivity analysis for ensemble predictions.

    Perturbs XGBoost feature inputs with Gaussian noise; Elo and Poisson
    are deterministic given team names and are not perturbed.
    """

    def __init__(
        self,
        elo_model: EloModel,
        poisson_model: PoissonModel,
        xgb_model: XGBoostMatchModel,
        feature_builder: FeatureBuilder,
        weights: tuple[float, float, float] = (0.3, 0.3, 0.4),
        config: DiagnosticsConfig | None = None,
        random_seed: int = 42,
    ) -> None:
        self._elo = elo_model
        self._poisson = poisson_model
        self._xgb = xgb_model
        self._feature_builder = feature_builder
        self._w_elo, self._w_poisson, self._w_xgb = weights
        cfg = config or settings.diagnostics
        self._n = cfg.stability_n_perturbations
        self._noise = cfg.stability_noise_scale
        self._rng = np.random.default_rng(random_seed)

    def analyze(
        self,
        home_team: str,
        away_team: str,
        match_date: pd.Timestamp | None = None,
    ) -> dict:
        """Run perturbation analysis for a match.

        Returns:
            dict with keys: home_team, away_team, base_probabilities,
            mean_probabilities, std_probabilities, stability_band, n_perturbations
        """
        # Elo probabilities (deterministic)
        ew, ed, el = self._elo.win_draw_loss_probabilities(home_team, away_team)
        elo_probs = np.array([ew, ed, el])  # [H, D, A]

        # Poisson probabilities (deterministic)
        pw, pd_, pl = self._poisson.win_draw_loss_from_poisson(home_team, away_team)
        poisson_probs = np.array([pw, pd_, pl])  # [H, D, A]

        # Resolve match_date — build_features_for_match requires a pd.Timestamp
        effective_date: pd.Timestamp = match_date if match_date is not None else pd.Timestamp.now()

        # Base features (unperturbed)
        try:
            base_features = self._feature_builder.build_features_for_match(
                home_team, away_team, effective_date
            )
        except Exception as exc:
            logger.warning(f"Could not build features for {home_team} vs {away_team}: {exc}")
            base_features = {}

        # Base XGBoost prediction
        base_xgb = self._xgb_probs(base_features)
        base_ensemble = self._blend(elo_probs, poisson_probs, base_xgb)

        # Perturbation runs
        perturbed_ensembles = []
        for _ in range(self._n):
            perturbed = self._perturb_features(base_features)
            xgb_p = self._xgb_probs(perturbed)
            ensemble_p = self._blend(elo_probs, poisson_probs, xgb_p)
            perturbed_ensembles.append(ensemble_p)

        arr = np.array(perturbed_ensembles)  # shape (n, 3) — [H, D, A]
        mean_probs = arr.mean(axis=0)
        std_probs = arr.std(axis=0)
        max_std = float(std_probs.max())

        band = "Unstable" if max_std >= 0.05 else ("Moderate" if max_std >= 0.02 else "Stable")

        return {
            "home_team": home_team,
            "away_team": away_team,
            "base_probabilities": {
                "home_win": round(float(base_ensemble[0]), 4),
                "draw": round(float(base_ensemble[1]), 4),
                "away_win": round(float(base_ensemble[2]), 4),
            },
            "mean_probabilities": {
                "home_win": round(float(mean_probs[0]), 4),
                "draw": round(float(mean_probs[1]), 4),
                "away_win": round(float(mean_probs[2]), 4),
            },
            "std_probabilities": {
                "home_win": round(float(std_probs[0]), 4),
                "draw": round(float(std_probs[1]), 4),
                "away_win": round(float(std_probs[2]), 4),
            },
            "stability_band": band,
            "n_perturbations": self._n,
        }

    def _xgb_probs(self, features: dict) -> np.ndarray:
        """Get XGBoost probs as [H, D, A] array. Returns uniform on failure."""
        try:
            df = pd.DataFrame([features])
            # Keep only known feature columns that exist in the DataFrame
            cols = [c for c in FEATURE_COLUMNS if c in df.columns]
            if not cols:
                return np.array([1 / 3, 1 / 3, 1 / 3])
            result = self._xgb.predict_proba_dict(df[cols])
            p = result[0] if isinstance(result, list) else result
            return np.array([p["home_win"], p["draw"], p["away_win"]])
        except Exception as exc:
            logger.debug(f"XGBoost prediction failed: {exc}")
            return np.array([1 / 3, 1 / 3, 1 / 3])

    def _perturb_features(self, features: dict) -> dict:
        """Add Gaussian noise (scaled by noise_scale) to numeric features."""
        perturbed = {}
        for k, v in features.items():
            if isinstance(v, (int, float)) and v is not None:
                noise = self._rng.normal(0, self._noise * (abs(v) + 1e-6))
                perturbed[k] = v + noise
            else:
                perturbed[k] = v
        return perturbed

    def _blend(
        self,
        elo: np.ndarray,
        poisson: np.ndarray,
        xgb: np.ndarray,
    ) -> np.ndarray:
        """Weighted blend, normalized."""
        blended = self._w_elo * elo + self._w_poisson * poisson + self._w_xgb * xgb
        total = blended.sum()
        return blended / total if total > 0 else blended
