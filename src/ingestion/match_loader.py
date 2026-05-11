from pathlib import Path

import pandas as pd
from loguru import logger

REQUIRED_COLUMNS = {"date", "home_team", "away_team", "home_goals", "away_goals"}


def load_matches(path: Path) -> pd.DataFrame:
    """Load and validate historical match data from CSV."""
    df = pd.read_csv(path, parse_dates=["date"])
    _validate(df, path)
    df = df.dropna(subset=["home_team", "away_team", "home_goals", "away_goals"])
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df = df[df["home_goals"] >= 0]
    df = df[df["away_goals"] >= 0]
    df = df.drop_duplicates(subset=["date", "home_team", "away_team"])
    df = df.sort_values("date").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} matches from {path.name}")
    return df


def _validate(df: pd.DataFrame, path: Path) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")

    null_counts = df[list(REQUIRED_COLUMNS)].isnull().sum()
    nulls = null_counts[null_counts > 0]
    if not nulls.empty:
        logger.warning(f"Null values in {path.name}: {nulls.to_dict()}")

    if "home_goals" in df.columns and "away_goals" in df.columns:
        invalid = df[(df["home_goals"] < 0) | (df["away_goals"] < 0)]
        if not invalid.empty:
            logger.warning(f"Dropping {len(invalid)} rows with negative scores")
