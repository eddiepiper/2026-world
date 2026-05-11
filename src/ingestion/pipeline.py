from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

DEFAULT_TEAM_MAP: dict[str, str] = {
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "USA": "United States",
    "Türkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
    "China PR": "China",
    "DR Congo": "Congo DR",
}


class DataPipeline:
    """Quality-checked ingestion pipeline for international football results."""

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def load_raw(self, path: Path) -> pd.DataFrame:
        """Load the raw results CSV as-is (no cleaning)."""
        df = pd.read_csv(path, parse_dates=["date"])
        logger.info(f"Loaded raw data: {len(df):,} rows from {path.name}")
        return df

    def save_processed(self, df: pd.DataFrame, path: Path) -> None:
        """Save the cleaned DataFrame to a CSV file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(df):,} processed rows → {path}")

    # ------------------------------------------------------------------
    # Quality checks
    # ------------------------------------------------------------------

    def quality_check(self, df: pd.DataFrame) -> dict:
        """Return a quality report without modifying the DataFrame.

        Report keys:
            total_rows        — number of rows
            missing_values    — per-column null counts (only columns with nulls)
            duplicates        — count of rows sharing (date, home_team, away_team)
            invalid_scores    — count of rows with negative or non-numeric scores
            team_name_issues  — count of team names in DEFAULT_TEAM_MAP
        """
        report: dict = {}

        report["total_rows"] = len(df)

        # Missing values per column (only report columns that have any)
        null_counts = df.isnull().sum()
        report["missing_values"] = {
            col: int(n) for col, n in null_counts.items() if n > 0
        }
        if report["missing_values"]:
            logger.warning(f"Missing values detected: {report['missing_values']}")

        # Duplicate match records
        dup_mask = df.duplicated(subset=["date", "home_team", "away_team"], keep=False)
        report["duplicates"] = int(dup_mask.sum())
        if report["duplicates"]:
            logger.warning(f"Found {report['duplicates']} rows involved in duplicate matches")

        # Invalid scores: non-numeric or negative
        invalid_count = 0
        for col in ("home_score", "away_score"):
            if col in df.columns:
                numeric_series = pd.to_numeric(df[col], errors="coerce")
                non_numeric = numeric_series.isnull() & df[col].notna()
                negative = numeric_series < 0
                bad = non_numeric | negative
                invalid_count += int(bad.sum())
        report["invalid_scores"] = invalid_count
        if invalid_count:
            logger.warning(f"Found {invalid_count} rows with invalid (negative or non-numeric) scores")

        # Team name issues (names that should be normalised)
        team_name_issues = 0
        for col in ("home_team", "away_team"):
            if col in df.columns:
                team_name_issues += int(df[col].isin(DEFAULT_TEAM_MAP).sum())
        report["team_name_issues"] = team_name_issues
        if team_name_issues:
            logger.warning(
                f"Found {team_name_issues} team name occurrences that should be standardised"
            )

        return report

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------

    def standardize_teams(
        self,
        df: pd.DataFrame,
        team_map: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Apply team name normalisation using team_map (defaults to DEFAULT_TEAM_MAP).

        Returns a new DataFrame — never mutates the input.
        """
        mapping = DEFAULT_TEAM_MAP if team_map is None else team_map
        out = df.copy()
        for col in ("home_team", "away_team"):
            if col in out.columns:
                out[col] = out[col].replace(mapping)
        applied = sum(
            int((df[col] != out[col]).sum())
            for col in ("home_team", "away_team")
            if col in df.columns
        )
        logger.info(f"Standardised {applied} team name occurrences")
        return out

    def clean(self, df: pd.DataFrame, start_year: int = 1980) -> pd.DataFrame:
        """Return a clean, analysis-ready DataFrame.

        Steps (each logged):
          1. Filter to start_year–present
          2. Rename home_score → home_goals, away_score → away_goals
          3. Drop rows with missing scores (with reason logged)
          4. Drop rows with negative scores (with reason logged)
          5. Drop duplicate (date, home_team, away_team) rows (logged)
          6. Sort chronologically
        """
        out = df.copy()
        initial = len(out)

        # 1. Year filter
        out = out[out["date"].dt.year >= start_year].copy()
        dropped_year = initial - len(out)
        if dropped_year:
            logger.info(f"Dropped {dropped_year:,} rows before {start_year}")

        # 2. Rename score columns
        rename_map = {}
        if "home_score" in out.columns:
            rename_map["home_score"] = "home_goals"
        if "away_score" in out.columns:
            rename_map["away_score"] = "away_goals"
        if rename_map:
            out = out.rename(columns=rename_map)

        # 3. Coerce goals to numeric
        for col in ("home_goals", "away_goals"):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        # 4. Drop missing scores
        before_null = len(out)
        out = out.dropna(subset=["home_goals", "away_goals"])
        dropped_null = before_null - len(out)
        if dropped_null:
            logger.warning(f"Dropped {dropped_null} rows with missing scores")

        # 5. Drop negative scores
        before_neg = len(out)
        invalid_mask = (out["home_goals"] < 0) | (out["away_goals"] < 0)
        if invalid_mask.any():
            logger.warning(
                f"Dropping {invalid_mask.sum()} rows with negative scores: "
                + str(out[invalid_mask][["date", "home_team", "away_team", "home_goals", "away_goals"]].head())
            )
        out = out[~invalid_mask]
        dropped_neg = before_neg - len(out)

        # 6. Drop duplicates
        before_dup = len(out)
        out = out.drop_duplicates(subset=["date", "home_team", "away_team"])
        dropped_dup = before_dup - len(out)
        if dropped_dup:
            logger.warning(f"Dropped {dropped_dup} duplicate match rows")

        # 7. Cast goals to int
        out["home_goals"] = out["home_goals"].astype(int)
        out["away_goals"] = out["away_goals"].astype(int)

        # 8. Chronological sort
        out = out.sort_values("date").reset_index(drop=True)

        logger.info(
            f"Clean complete: {initial:,} → {len(out):,} rows "
            f"(dropped {initial - len(out):,} total)"
        )
        return out

    # ------------------------------------------------------------------
    # Coverage logging
    # ------------------------------------------------------------------

    def log_coverage(self, df: pd.DataFrame) -> None:
        """Log dataset coverage statistics."""
        if df.empty:
            logger.warning("DataFrame is empty — no coverage to report")
            return

        min_year = df["date"].dt.year.min()
        max_year = df["date"].dt.year.max()
        total_matches = len(df)

        all_teams: set[str] = set()
        for col in ("home_team", "away_team"):
            if col in df.columns:
                all_teams.update(df[col].dropna().unique())

        logger.info(f"Coverage: {min_year}–{max_year}")
        logger.info(f"Total matches: {total_matches:,}")
        logger.info(f"Unique teams: {len(all_teams):,}")

        if "tournament" in df.columns:
            tourney_counts = df["tournament"].value_counts()
            unique_tourneys = len(tourney_counts)
            top10 = tourney_counts.head(10)
            logger.info(f"Tournaments: {unique_tourneys} unique")
            logger.info(f"Top 10 by volume:\n{top10.to_string()}")

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(
        self,
        raw_path: Path,
        processed_path: Path,
        start_year: int = 1980,
    ) -> pd.DataFrame:
        """Full pipeline: load → quality_check → standardize → clean → log → save."""
        df = self.load_raw(raw_path)
        self.quality_check(df)
        df = self.standardize_teams(df)
        df = self.clean(df, start_year=start_year)
        self.log_coverage(df)
        self.save_processed(df, processed_path)
        return df
