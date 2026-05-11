from pathlib import Path

import pandas as pd
from loguru import logger

REQUIRED_COLUMNS = {"team_name", "confederation", "country_code"}


def load_teams(path: Path) -> pd.DataFrame:
    """Load and validate team metadata from CSV."""
    df = pd.read_csv(path)
    _validate(df, path)
    df = df.dropna(subset=["team_name"])
    df = df.drop_duplicates(subset=["team_name"])
    logger.info(f"Loaded {len(df)} teams from {path.name}")
    return df


def _validate(df: pd.DataFrame, path: Path) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")
