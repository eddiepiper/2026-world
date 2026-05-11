"""Tests for src/features/ — rolling form, team strength, and feature builder."""
from __future__ import annotations

import pandas as pd
import pytest

from src.features.rolling_form import RollingFormCalculator
from src.features.team_strength import TeamStrengthFeatures
from src.features.feature_builder import FeatureBuilder, FEATURE_COLUMNS
from src.models.elo_model import EloModel
from src.models.poisson_model import PoissonModel


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_matches(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal matches DataFrame from a list of row dicts."""
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


SAMPLE_MATCHES = [
    {"date": "2020-01-01", "home_team": "A", "away_team": "B", "home_goals": 2, "away_goals": 1},
    {"date": "2020-01-08", "home_team": "B", "away_team": "A", "home_goals": 0, "away_goals": 0},
    {"date": "2020-01-15", "home_team": "A", "away_team": "C", "home_goals": 3, "away_goals": 0},
    {"date": "2020-01-22", "home_team": "C", "away_team": "B", "home_goals": 1, "away_goals": 2},
    {"date": "2020-01-29", "home_team": "A", "away_team": "B", "home_goals": 1, "away_goals": 2},
    {"date": "2020-02-05", "home_team": "B", "away_team": "C", "home_goals": 2, "away_goals": 2},
]


@pytest.fixture()
def matches_df() -> pd.DataFrame:
    return _make_matches(SAMPLE_MATCHES)


@pytest.fixture()
def calculator(matches_df: pd.DataFrame) -> RollingFormCalculator:
    return RollingFormCalculator(matches_df, window=5)


@pytest.fixture()
def trained_elo(matches_df: pd.DataFrame) -> EloModel:
    model = EloModel()
    model.train_on_matches(matches_df)
    return model


@pytest.fixture()
def trained_poisson(matches_df: pd.DataFrame) -> PoissonModel:
    model = PoissonModel()
    model.fit(matches_df)
    return model


@pytest.fixture()
def feature_builder(matches_df, trained_elo, trained_poisson) -> FeatureBuilder:
    return FeatureBuilder(
        matches_df=matches_df,
        elo_model=trained_elo,
        poisson_model=trained_poisson,
    )


# ---------------------------------------------------------------------------
# RollingFormCalculator tests
# ---------------------------------------------------------------------------

class TestRollingFormWinRateEmptyHistory:
    def test_returns_zero_for_unknown_team(self, calculator):
        result = calculator.win_rate("Unknown", pd.Timestamp("2020-01-01"))
        assert result == 0.0

    def test_returns_zero_before_any_match(self, calculator):
        # Team A's first match is 2020-01-01; query before that date → 0.0
        result = calculator.win_rate("A", pd.Timestamp("2019-12-31"))
        assert result == 0.0


class TestRollingFormWinRateWithData:
    def test_correct_win_rate(self, calculator):
        # Before 2020-02-05 team A has played:
        # 2020-01-01: A 2-1 B → win
        # 2020-01-08: B 0-0 A → draw (not a win)
        # 2020-01-15: A 3-0 C → win
        # 2020-01-29: A 1-2 B → loss
        # window=5 → 2 wins out of 4 matches → 0.5
        rate = calculator.win_rate("A", pd.Timestamp("2020-02-05"))
        assert rate == pytest.approx(0.5)

    def test_win_rate_is_between_0_and_1(self, calculator):
        for date in ["2020-01-10", "2020-01-20", "2020-02-10"]:
            rate = calculator.win_rate("B", pd.Timestamp(date))
            assert 0.0 <= rate <= 1.0


class TestRollingFormGoalsScoredAvg:
    def test_goals_scored_avg_no_history(self, calculator):
        assert calculator.goals_scored_avg("A", pd.Timestamp("2019-01-01")) == 0.0

    def test_goals_scored_avg_correct(self, calculator):
        # Team A before 2020-02-05:
        # 2020-01-01 home: 2 goals scored
        # 2020-01-08 away: 0 goals scored
        # 2020-01-15 home: 3 goals scored
        # 2020-01-29 home: 1 goals scored
        # window=5 → (2+0+3+1)/4 = 1.5
        avg = calculator.goals_scored_avg("A", pd.Timestamp("2020-02-05"))
        assert avg == pytest.approx(1.5)


class TestRollingFormCleanSheetRate:
    def test_clean_sheet_rate_no_history(self, calculator):
        assert calculator.clean_sheet_rate("A", pd.Timestamp("2019-01-01")) == 0.0

    def test_clean_sheet_rate_correct(self, calculator):
        # Team A before 2020-02-05:
        # 2020-01-01: conceded 1 → not clean
        # 2020-01-08: conceded 0 → clean
        # 2020-01-15: conceded 0 → clean
        # 2020-01-29: conceded 2 → not clean
        # window=5 → 2 clean / 4 = 0.5
        rate = calculator.clean_sheet_rate("A", pd.Timestamp("2020-02-05"))
        assert rate == pytest.approx(0.5)


class TestRollingFormDaysSinceLastMatch:
    def test_no_prior_match_returns_30(self, calculator):
        assert calculator.days_since_last_match("A", pd.Timestamp("2019-01-01")) == 30.0

    def test_days_since_correct(self, calculator):
        # A's last match before 2020-02-05 is 2020-01-29 → 7 days
        days = calculator.days_since_last_match("A", pd.Timestamp("2020-02-05"))
        assert days == pytest.approx(7.0)


class TestRollingFormNoLeakage:
    """Verify that statistics at date T do not include the match played on T."""

    def test_win_rate_excludes_same_day_match(self, matches_df):
        calc = RollingFormCalculator(matches_df, window=5)
        # Match on 2020-01-01: A beats B 2-1.
        # Query win_rate for A *at* 2020-01-01 — must NOT include that match.
        rate_at = calc.win_rate("A", pd.Timestamp("2020-01-01"))
        rate_before = calc.win_rate("A", pd.Timestamp("2019-12-31"))
        assert rate_at == rate_before  # both should be 0.0 (no prior matches)

    def test_goals_scored_excludes_same_day(self, matches_df):
        calc = RollingFormCalculator(matches_df, window=5)
        # Query right at the date of A's first match — should still return 0.0
        avg = calc.goals_scored_avg("A", pd.Timestamp("2020-01-01"))
        assert avg == 0.0


# ---------------------------------------------------------------------------
# TeamStrengthFeatures tests
# ---------------------------------------------------------------------------

class TestTeamStrengthEloDiff:
    def test_elo_diff_same_team_is_zero(self, trained_elo, trained_poisson):
        tsf = TeamStrengthFeatures(trained_elo, trained_poisson)
        # A team vs itself must have diff 0
        assert tsf.elo_diff("A", "A") == pytest.approx(0.0)

    def test_elo_diff_sign(self, trained_elo, trained_poisson):
        tsf = TeamStrengthFeatures(trained_elo, trained_poisson)
        diff_ab = tsf.elo_diff("A", "B")
        diff_ba = tsf.elo_diff("B", "A")
        assert diff_ab == pytest.approx(-diff_ba)


class TestTeamStrengthFifaDiffMissingTeam:
    def test_default_fifa_points_for_missing_team(self, trained_elo, trained_poisson):
        tsf = TeamStrengthFeatures(trained_elo, trained_poisson, rankings_df=None)
        assert tsf.fifa_points("UnknownTeam") == 1000.0

    def test_fifa_diff_missing_teams_is_zero(self, trained_elo, trained_poisson):
        tsf = TeamStrengthFeatures(trained_elo, trained_poisson, rankings_df=None)
        assert tsf.fifa_points_diff("X", "Y") == pytest.approx(0.0)

    def test_fifa_points_from_rankings(self, trained_elo, trained_poisson):
        rankings = pd.DataFrame({"team": ["A", "B"], "points": [1500.0, 1200.0]})
        tsf = TeamStrengthFeatures(trained_elo, trained_poisson, rankings_df=rankings)
        assert tsf.fifa_points("A") == pytest.approx(1500.0)
        assert tsf.fifa_points_diff("A", "B") == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# FeatureBuilder tests
# ---------------------------------------------------------------------------

class TestFeatureBuilderBuildFeaturesForMatch:
    def test_has_all_required_columns(self, feature_builder):
        feats = feature_builder.build_features_for_match(
            "A", "B", pd.Timestamp("2020-03-01")
        )
        for col in FEATURE_COLUMNS:
            assert col in feats, f"Missing feature column: {col}"

    def test_all_values_are_float(self, feature_builder):
        feats = feature_builder.build_features_for_match(
            "A", "B", pd.Timestamp("2020-03-01")
        )
        for col in FEATURE_COLUMNS:
            assert isinstance(feats[col], float), (
                f"Column '{col}' is {type(feats[col])}, expected float"
            )

    def test_is_neutral_flag(self, feature_builder):
        feats_home = feature_builder.build_features_for_match(
            "A", "B", pd.Timestamp("2020-03-01"), is_neutral=False
        )
        feats_neutral = feature_builder.build_features_for_match(
            "A", "B", pd.Timestamp("2020-03-01"), is_neutral=True
        )
        assert feats_home["is_neutral"] == 0.0
        assert feats_neutral["is_neutral"] == 1.0


class TestFeatureBuilderTrainingMatrixLabelCorrectness:
    def test_label_values_are_valid(self, feature_builder):
        matrix = feature_builder.build_training_matrix()
        valid_labels = {"H", "D", "A"}
        assert set(matrix["label"].unique()).issubset(valid_labels)

    def test_known_match_label(self, feature_builder, matches_df):
        matrix = feature_builder.build_training_matrix()
        # First match: A 2-1 B → H
        assert matrix.iloc[0]["label"] == "H"

    def test_draw_label(self, feature_builder):
        # Match 2020-01-08: B 0-0 A → D
        matrix = feature_builder.build_training_matrix()
        row = matrix.iloc[1]
        assert row["label"] == "D"

    def test_away_win_label(self, feature_builder):
        # Match 2020-01-29: A 1-2 B → A
        matrix = feature_builder.build_training_matrix()
        row = matrix.iloc[4]
        assert row["label"] == "A"


class TestFeatureBuilderTrainingMatrixNoNanLabels:
    def test_no_nan_labels(self, feature_builder):
        matrix = feature_builder.build_training_matrix()
        assert matrix["label"].isna().sum() == 0

    def test_no_nan_features(self, feature_builder):
        matrix = feature_builder.build_training_matrix()
        feature_cols = [c for c in matrix.columns if c != "label"]
        assert matrix[feature_cols].isna().sum().sum() == 0

    def test_row_count_matches_input(self, feature_builder, matches_df):
        matrix = feature_builder.build_training_matrix()
        assert len(matrix) == len(matches_df)


class TestFeatureBuilderNoLeakage:
    """Verify that rolling form at match i is computed BEFORE that match."""

    def test_first_match_has_zero_rest_stats(self, matches_df, trained_elo, trained_poisson):
        """For the very first match (no prior history), rolling stats must be 0."""
        builder = FeatureBuilder(
            matches_df=matches_df,
            elo_model=trained_elo,
            poisson_model=trained_poisson,
        )
        matrix = builder.build_training_matrix()
        first = matrix.iloc[0]
        # No prior matches for either team → win rates and goals averages = 0
        assert first["home_win_rate"] == 0.0
        assert first["away_win_rate"] == 0.0
        assert first["home_goals_scored"] == 0.0
        assert first["away_goals_scored"] == 0.0

    def test_elo_increases_after_first_win(self, matches_df, trained_elo, trained_poisson):
        """After the first match (A beats B), A's home_elo should be higher in row 1."""
        builder = FeatureBuilder(
            matches_df=matches_df,
            elo_model=trained_elo,
            poisson_model=trained_poisson,
        )
        matrix = builder.build_training_matrix()
        # Row 0: home=A, away=B → A starts at default (ratings equal)
        # Row 1: home=B, away=A → A's rating should now reflect the win in row 0
        row0_home_elo = matrix.iloc[0]["home_elo"]
        row1_away_elo = matrix.iloc[1]["away_elo"]  # A is away in match 1
        # A won match 0, so row1 away_elo (A) should be > row0 home_elo (A at initial)
        assert row1_away_elo > row0_home_elo

    def test_rolling_form_excludes_current_match(self, matches_df, trained_elo, trained_poisson):
        """Stats at match i must not reflect the outcome of match i."""
        # Match index 2 is 2020-01-15: A vs C (A wins 3-0).
        # home_win_rate for A at that point should be based on matches 0 and 1 only.
        builder = FeatureBuilder(
            matches_df=matches_df,
            elo_model=trained_elo,
            poisson_model=trained_poisson,
        )
        matrix = builder.build_training_matrix()
        # A at match 2 (index 2): prior matches are rows 0 (win) and 1 (draw as away).
        # window=5 → 1 win / 2 matches → 0.5
        row2 = matrix.iloc[2]
        assert row2["home_win_rate"] == pytest.approx(0.5)
