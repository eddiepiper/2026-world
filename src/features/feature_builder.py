"""Feature matrix builder — orchestrates rolling form + strength features.

The golden rule is enforced here:
    Features for match M use ONLY data from matches *before* M's date.

For ``build_training_matrix`` the Elo model is replayed chronologically so
that ratings at match time reflect only the matches already processed.
"""
from __future__ import annotations

import pandas as pd
from loguru import logger

from src.config.settings import EloConfig
from src.models.elo_model import EloModel
from src.models.poisson_model import PoissonModel
from src.features.rolling_form import RollingFormCalculator
from src.features.team_strength import TeamStrengthFeatures


# Required feature column names (order preserved for readability)
FEATURE_COLUMNS: list[str] = [
    "elo_diff",
    "home_elo",
    "away_elo",
    "fifa_diff",
    "home_attack",
    "home_defense",
    "away_attack",
    "away_defense",
    "home_win_rate",
    "away_win_rate",
    "home_goals_scored",
    "home_goals_conceded",
    "away_goals_scored",
    "away_goals_conceded",
    "home_clean_sheet_rate",
    "away_clean_sheet_rate",
    "home_days_rest",
    "away_days_rest",
    "is_neutral",
]


class FeatureBuilder:
    """Build ML-ready feature matrices from historical match data."""

    def __init__(
        self,
        matches_df: pd.DataFrame,
        elo_model: EloModel,
        poisson_model: PoissonModel,
        rankings_df: pd.DataFrame | None = None,
        form_window: int = 5,
    ) -> None:
        """
        Parameters
        ----------
        matches_df:
            Full historical match DataFrame with columns:
            date, home_team, away_team, home_goals, away_goals.
            Optionally includes ``neutral`` (bool/int) for venue info.
        elo_model:
            A trained ``EloModel`` instance used for inference features.
            *Not* used during ``build_training_matrix`` (a fresh model is
            replayed chronologically there to prevent leakage).
        poisson_model:
            A trained ``PoissonModel`` instance.
        rankings_df:
            Optional DataFrame with columns ``[team, points]``.
        form_window:
            Look-back window for rolling form statistics (default 5).
        """
        required = {"date", "home_team", "away_team", "home_goals", "away_goals"}
        missing = required - set(matches_df.columns)
        if missing:
            raise ValueError(f"matches_df missing columns: {missing}")

        self._matches = matches_df.copy()
        self._matches["date"] = pd.to_datetime(self._matches["date"])

        self._elo_model = elo_model
        self._poisson_model = poisson_model
        self._rankings_df = rankings_df
        self._form_window = form_window

        # Helpers re-used for single-match inference
        self._form = RollingFormCalculator(self._matches, window=form_window)
        self._strength = TeamStrengthFeatures(
            elo_model=elo_model,
            poisson_model=poisson_model,
            rankings_df=rankings_df,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_feature_dict(
        self,
        home_team: str,
        away_team: str,
        match_date: pd.Timestamp,
        strength: TeamStrengthFeatures,
        form: RollingFormCalculator,
        is_neutral: float = 0.0,
    ) -> dict[str, float]:
        """Assemble the feature dict given pre-built helper objects."""
        match_date = pd.Timestamp(match_date)
        return {
            "elo_diff": strength.elo_diff(home_team, away_team),
            "home_elo": strength.elo_rating(home_team),
            "away_elo": strength.elo_rating(away_team),
            "fifa_diff": strength.fifa_points_diff(home_team, away_team),
            "home_attack": strength.attack_strength(home_team),
            "home_defense": strength.defense_strength(home_team),
            "away_attack": strength.attack_strength(away_team),
            "away_defense": strength.defense_strength(away_team),
            "home_win_rate": form.win_rate(home_team, match_date),
            "away_win_rate": form.win_rate(away_team, match_date),
            "home_goals_scored": form.goals_scored_avg(home_team, match_date),
            "home_goals_conceded": form.goals_conceded_avg(home_team, match_date),
            "away_goals_scored": form.goals_scored_avg(away_team, match_date),
            "away_goals_conceded": form.goals_conceded_avg(away_team, match_date),
            "home_clean_sheet_rate": form.clean_sheet_rate(home_team, match_date),
            "away_clean_sheet_rate": form.clean_sheet_rate(away_team, match_date),
            "home_days_rest": form.days_since_last_match(home_team, match_date),
            "away_days_rest": form.days_since_last_match(away_team, match_date),
            "is_neutral": is_neutral,
        }

    @staticmethod
    def _result_label(home_goals: int, away_goals: int) -> str:
        if home_goals > away_goals:
            return "H"
        if home_goals == away_goals:
            return "D"
        return "A"

    @staticmethod
    def _neutral_flag(row: pd.Series) -> float:
        if "neutral" in row.index:
            return 1.0 if bool(row["neutral"]) else 0.0
        return 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_features_for_match(
        self,
        home_team: str,
        away_team: str,
        match_date: pd.Timestamp,
        is_neutral: bool = False,
    ) -> dict[str, float]:
        """Build feature dict for a single match (inference on future matches).

        Uses the pre-trained Elo and Poisson models passed at construction
        time, plus rolling statistics from all historical matches before
        *match_date*.
        """
        return self._build_feature_dict(
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            strength=self._strength,
            form=self._form,
            is_neutral=1.0 if is_neutral else 0.0,
        )

    def build_training_matrix(self) -> pd.DataFrame:
        """Build feature matrix + target label for ALL matches in matches_df.

        Returns a DataFrame with columns:
            [all feature columns..., 'label']

        label values: 'H' (home win), 'D' (draw), 'A' (away win).

        CRITICAL no-leakage guarantee
        --------------------------------
        A fresh ``EloModel`` is created and replayed chronologically.
        For match i (0-indexed) the Elo ratings reflect only matches 0..i-1.
        Rolling form uses ``date < match_date`` (strict less-than).
        """
        sorted_matches = self._matches.sort_values("date").reset_index(drop=True)

        # Fresh Elo model — ratings start at initial_rating for everyone
        replay_elo = EloModel(config=self._elo_model.config)

        # Poisson model is already fitted on historical data; because attack/
        # defense strengths are global averages they don't leak individual
        # match outcomes (they're computed once before any single prediction).
        replay_form = RollingFormCalculator(self._matches, window=self._form_window)
        replay_strength = TeamStrengthFeatures(
            elo_model=replay_elo,
            poisson_model=self._poisson_model,
            rankings_df=self._rankings_df,
        )

        rows: list[dict[str, float | str]] = []
        total = len(sorted_matches)
        log_step = max(1, total // 10)

        for i, row in sorted_matches.iterrows():
            if i % log_step == 0:
                logger.debug("build_training_matrix: {}/{}", i, total)

            home_team: str = row["home_team"]
            away_team: str = row["away_team"]
            match_date: pd.Timestamp = pd.Timestamp(row["date"])
            home_goals: int = int(row["home_goals"])
            away_goals: int = int(row["away_goals"])
            is_neutral: float = self._neutral_flag(row)

            # --- compute features BEFORE updating ratings ---
            feat = self._build_feature_dict(
                home_team=home_team,
                away_team=away_team,
                match_date=match_date,
                strength=replay_strength,
                form=replay_form,
                is_neutral=is_neutral,
            )
            feat["label"] = self._result_label(home_goals, away_goals)
            rows.append(feat)

            # --- update Elo ratings AFTER features are captured ---
            replay_elo.update_ratings(home_team, away_team, home_goals, away_goals)

        df = pd.DataFrame(rows, columns=FEATURE_COLUMNS + ["label"])
        logger.info(
            "build_training_matrix complete: {} rows, {} features + label",
            len(df),
            len(FEATURE_COLUMNS),
        )
        return df
