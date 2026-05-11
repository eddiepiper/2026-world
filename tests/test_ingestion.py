import pandas as pd
import pytest
from pathlib import Path

from src.ingestion.match_loader import load_matches
from src.ingestion.team_loader import load_teams
from src.ingestion.fifa_loader import load_fifa_rankings

SAMPLE = Path(__file__).parent.parent / "data" / "sample"


# --- match_loader ---

def test_load_matches_returns_dataframe():
    df = load_matches(SAMPLE / "matches.csv")
    assert isinstance(df, pd.DataFrame)


def test_load_matches_has_required_columns():
    df = load_matches(SAMPLE / "matches.csv")
    for col in ["date", "home_team", "away_team", "home_goals", "away_goals"]:
        assert col in df.columns


def test_load_matches_no_nulls_in_key_columns():
    df = load_matches(SAMPLE / "matches.csv")
    assert df[["home_team", "away_team", "home_goals", "away_goals"]].isnull().sum().sum() == 0


def test_load_matches_goals_non_negative():
    df = load_matches(SAMPLE / "matches.csv")
    assert (df["home_goals"] >= 0).all()
    assert (df["away_goals"] >= 0).all()


def test_load_matches_no_duplicates():
    df = load_matches(SAMPLE / "matches.csv")
    dupes = df.duplicated(subset=["date", "home_team", "away_team"])
    assert not dupes.any()


def test_load_matches_sorted_by_date():
    df = load_matches(SAMPLE / "matches.csv")
    assert (df["date"].diff().dropna() >= pd.Timedelta(0)).all()


def test_load_matches_raises_on_missing_column(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("date,home_team,away_team\n2022-01-01,A,B\n")
    with pytest.raises(ValueError, match="Missing columns"):
        load_matches(bad_csv)


# --- team_loader ---

def test_load_teams_returns_dataframe():
    df = load_teams(SAMPLE / "teams.csv")
    assert isinstance(df, pd.DataFrame)


def test_load_teams_has_required_columns():
    df = load_teams(SAMPLE / "teams.csv")
    for col in ["team_name", "confederation", "country_code"]:
        assert col in df.columns


def test_load_teams_no_duplicate_names():
    df = load_teams(SAMPLE / "teams.csv")
    assert not df["team_name"].duplicated().any()


def test_load_teams_raises_on_missing_column(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("team_name\nBrazil\n")
    with pytest.raises(ValueError, match="Missing columns"):
        load_teams(bad_csv)


# --- fifa_loader ---

def test_load_fifa_rankings_returns_dataframe():
    df = load_fifa_rankings(SAMPLE / "fifa_rankings.csv")
    assert isinstance(df, pd.DataFrame)


def test_load_fifa_rankings_has_required_columns():
    df = load_fifa_rankings(SAMPLE / "fifa_rankings.csv")
    for col in ["rank", "team", "points"]:
        assert col in df.columns


def test_load_fifa_rankings_sorted_by_rank():
    df = load_fifa_rankings(SAMPLE / "fifa_rankings.csv")
    assert (df["rank"].diff().dropna() > 0).all()


def test_load_fifa_rankings_no_duplicate_teams():
    df = load_fifa_rankings(SAMPLE / "fifa_rankings.csv")
    assert not df["team"].duplicated().any()


def test_load_fifa_rankings_raises_on_missing_column(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("rank,team\n1,Brazil\n")
    with pytest.raises(ValueError, match="Missing columns"):
        load_fifa_rankings(bad_csv)
