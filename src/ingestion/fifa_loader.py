from pathlib import Path

import pandas as pd
from loguru import logger

REQUIRED_COLUMNS = {"rank", "team", "points"}


def load_fifa_rankings(path: Path) -> pd.DataFrame:
    """Load and validate FIFA ranking data from CSV."""
    df = pd.read_csv(path)
    _validate(df, path)
    df = df.dropna(subset=["rank", "team", "points"])
    df["rank"] = df["rank"].astype(int)
    df["points"] = df["points"].astype(float)
    df = df[df["rank"] >= 1]
    df = df.drop_duplicates(subset=["team"])
    df = df.sort_values("rank").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} FIFA rankings from {path.name}")
    return df


def _validate(df: pd.DataFrame, path: Path) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")
