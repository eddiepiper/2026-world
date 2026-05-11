"""Tests for src/ingestion/pipeline.py"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.pipeline import DataPipeline, DEFAULT_TEAM_MAP


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def pipeline() -> DataPipeline:
    return DataPipeline()


def _make_df(**overrides) -> pd.DataFrame:
    """Build a minimal valid results DataFrame for testing."""
    base = {
        "date": pd.to_datetime(["2000-01-01", "2001-06-15", "2010-07-11"]),
        "home_team": ["Brazil", "Germany", "Spain"],
        "away_team": ["Argentina", "France", "Netherlands"],
        "home_score": [3, 2, 1],
        "away_score": [0, 1, 0],
        "tournament": ["Friendly", "Friendly", "FIFA World Cup"],
        "city": ["Rio", "Berlin", "Johannesburg"],
        "country": ["Brazil", "Germany", "South Africa"],
        "neutral": [False, False, True],
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# quality_check
# ---------------------------------------------------------------------------

class TestQualityCheck:
    def test_quality_check_detects_duplicates(self, pipeline):
        df = _make_df()
        # Add an exact duplicate of row 0
        duplicate_row = df.iloc[[0]].copy()
        df = pd.concat([df, duplicate_row], ignore_index=True)
        report = pipeline.quality_check(df)
        assert report["duplicates"] >= 2  # both rows of the dupe pair are flagged

    def test_quality_check_detects_invalid_scores(self, pipeline):
        df = _make_df()
        df.loc[0, "home_score"] = -1  # negative score
        report = pipeline.quality_check(df)
        assert report["invalid_scores"] >= 1

    def test_quality_check_detects_missing_values(self, pipeline):
        df = _make_df()
        df.loc[1, "home_score"] = None
        report = pipeline.quality_check(df)
        assert "home_score" in report["missing_values"]
        assert report["missing_values"]["home_score"] >= 1

    def test_quality_check_returns_total_rows(self, pipeline):
        df = _make_df()
        report = pipeline.quality_check(df)
        assert report["total_rows"] == len(df)

    def test_quality_check_clean_data_has_zero_issues(self, pipeline):
        df = _make_df()
        report = pipeline.quality_check(df)
        assert report["duplicates"] == 0
        assert report["invalid_scores"] == 0
        assert report["missing_values"] == {}
        assert report["team_name_issues"] == 0


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------

class TestClean:
    def test_clean_filters_by_year(self, pipeline):
        df = _make_df(
            date=pd.to_datetime(["1970-01-01", "1995-06-01", "2010-07-11"]),
        )
        result = pipeline.clean(df, start_year=1990)
        assert (result["date"].dt.year >= 1990).all()
        assert len(result) == 2

    def test_clean_renames_columns(self, pipeline):
        df = _make_df()
        result = pipeline.clean(df, start_year=1990)
        assert "home_goals" in result.columns
        assert "away_goals" in result.columns
        assert "home_score" not in result.columns
        assert "away_score" not in result.columns

    def test_clean_drops_missing_scores(self, pipeline):
        df = _make_df()
        df.loc[0, "home_score"] = None
        result = pipeline.clean(df, start_year=1990)
        # The row with missing score should be gone
        assert result["home_goals"].isnull().sum() == 0

    def test_clean_drops_negative_scores(self, pipeline):
        df = _make_df()
        df.loc[0, "home_score"] = -1
        result = pipeline.clean(df, start_year=1990)
        assert (result["home_goals"] >= 0).all()

    def test_clean_drops_duplicate_matches(self, pipeline):
        df = _make_df()
        dup = df.iloc[[0]].copy()
        df = pd.concat([df, dup], ignore_index=True)
        result = pipeline.clean(df, start_year=1990)
        dupes = result.duplicated(subset=["date", "home_team", "away_team"])
        assert not dupes.any()

    def test_clean_sorts_chronologically(self, pipeline):
        df = _make_df(
            date=pd.to_datetime(["2010-07-11", "2000-01-01", "2001-06-15"]),
        )
        result = pipeline.clean(df, start_year=1990)
        diffs = result["date"].diff().dropna()
        assert (diffs >= pd.Timedelta(0)).all()

    def test_clean_goals_are_integers(self, pipeline):
        df = _make_df()
        result = pipeline.clean(df, start_year=1990)
        assert result["home_goals"].dtype == int
        assert result["away_goals"].dtype == int


# ---------------------------------------------------------------------------
# standardize_teams
# ---------------------------------------------------------------------------

class TestStandardizeTeams:
    def test_standardize_teams_applies_map(self, pipeline):
        df = _make_df(
            home_team=["Korea Republic", "Brazil", "IR Iran"],
            away_team=["Japan", "USA", "Germany"],
        )
        result = pipeline.standardize_teams(df)
        assert "South Korea" in result["home_team"].values
        assert "Korea Republic" not in result["home_team"].values
        assert "Iran" in result["home_team"].values
        assert "IR Iran" not in result["home_team"].values
        assert "United States" in result["away_team"].values
        assert "USA" not in result["away_team"].values

    def test_standardize_teams_custom_map(self, pipeline):
        df = _make_df(
            home_team=["England", "Germany", "France"],
            away_team=["Spain", "Italy", "Brazil"],
        )
        custom_map = {"England": "ENG", "Germany": "GER"}
        result = pipeline.standardize_teams(df, team_map=custom_map)
        assert "ENG" in result["home_team"].values
        assert "GER" in result["home_team"].values
        assert "England" not in result["home_team"].values

    def test_standardize_teams_does_not_mutate_input(self, pipeline):
        df = _make_df(
            home_team=["Korea Republic", "Brazil", "IR Iran"],
            away_team=["Japan", "USA", "Germany"],
        )
        original_home = df["home_team"].tolist()
        _ = pipeline.standardize_teams(df)
        assert df["home_team"].tolist() == original_home

    def test_standardize_teams_all_map_entries_covered(self, pipeline):
        """Every key in DEFAULT_TEAM_MAP should be replaced when present."""
        home_teams = list(DEFAULT_TEAM_MAP.keys())
        away_teams = ["Japan"] * len(home_teams)
        dates = pd.to_datetime(["2000-01-01"] * len(home_teams))
        df = pd.DataFrame({
            "date": dates,
            "home_team": home_teams,
            "away_team": away_teams,
            "home_score": [1] * len(home_teams),
            "away_score": [0] * len(home_teams),
            "tournament": ["Friendly"] * len(home_teams),
            "city": ["City"] * len(home_teams),
            "country": ["Country"] * len(home_teams),
            "neutral": [False] * len(home_teams),
        })
        result = pipeline.standardize_teams(df)
        for old_name in DEFAULT_TEAM_MAP:
            assert old_name not in result["home_team"].values, f"{old_name!r} was not replaced"


# ---------------------------------------------------------------------------
# log_coverage
# ---------------------------------------------------------------------------

class TestCoverageLogging:
    def test_coverage_logging_runs_without_error(self, pipeline):
        """Just ensure log_coverage does not raise."""
        df = _make_df()
        # Rename score cols to goals so log_coverage gets a realistic df
        df = df.rename(columns={"home_score": "home_goals", "away_score": "away_goals"})
        pipeline.log_coverage(df)  # must not raise

    def test_coverage_logging_empty_df_does_not_raise(self, pipeline):
        df = pd.DataFrame(columns=["date", "home_team", "away_team", "home_goals", "away_goals", "tournament"])
        df["date"] = pd.to_datetime(df["date"])
        pipeline.log_coverage(df)  # must not raise
