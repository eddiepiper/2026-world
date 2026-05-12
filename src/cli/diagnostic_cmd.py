from __future__ import annotations

import sys

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config.settings import settings
from src.diagnostics.confidence_scorer import ConfidenceScorer
from src.ensemble.ensemble_engine import EnsembleEngine
from src.features.feature_builder import FeatureBuilder
from src.ingestion.match_loader import load_matches
from src.ml.xgboost_model import XGBoostMatchModel
from src.models.elo_model import EloModel
from src.models.poisson_model import PoissonModel

console = Console()


def _load_ece_from_benchmark() -> float:
    """Load ECE for XGBoost from benchmark_results.csv, default 0.10 if unavailable.

    The CSV has columns: model, accuracy, log_loss, brier_score, rps, ece
    The XGBoost row is identified by 'model' column containing 'XGBoost'.
    """
    import pandas as pd

    ece_path = settings.outputs_dir / "evaluation" / "benchmark_results.csv"
    if not ece_path.exists():
        logger.warning("benchmark_results.csv not found; using default ECE=0.10")
        return 0.10
    try:
        df = pd.read_csv(ece_path)
        # Find model name column — may be 'model', 'model_name', etc.
        name_col = next((c for c in df.columns if "model" in c.lower()), None)
        # Find ECE column — named 'ece' or containing 'calibration'
        ece_col = next(
            (c for c in df.columns if c.lower() == "ece" or "calibration" in c.lower()),
            None,
        )
        if name_col and ece_col:
            xgb_row = df[df[name_col].str.contains("XGBoost", case=False, na=False)]
            if not xgb_row.empty:
                return float(xgb_row.iloc[0][ece_col])
        logger.warning(
            "Could not locate XGBoost ECE in benchmark_results.csv; using default 0.10"
        )
    except Exception as exc:
        logger.warning(f"Could not read ECE from benchmark_results.csv: {exc}")
    return 0.10


def cmd_confidence(home_team: str, away_team: str) -> None:
    """Compute and display confidence score for a match prediction."""
    model_path = settings.outputs_dir / "models" / "model.pkl"
    if not model_path.exists():
        console.print(
            "[bold red]No trained model found.[/bold red] "
            "Run [bold]python main.py train[/bold] first."
        )
        return

    data_path = settings.data_dir / "processed" / "matches.csv"
    if not data_path.exists():
        data_path = settings.data_dir / "sample" / "matches.csv"
    matches_df = load_matches(data_path)

    elo = EloModel(config=settings.elo)
    elo.train_on_matches(matches_df)
    poisson = PoissonModel(config=settings.poisson)
    poisson.fit(matches_df)
    xgb_model = XGBoostMatchModel.load(model_path)
    feature_builder = FeatureBuilder(
        matches_df=matches_df,
        elo_model=elo,
        poisson_model=poisson,
    )

    ensemble = EnsembleEngine(
        elo_model=elo,
        poisson_model=poisson,
        xgb_model=xgb_model,
        feature_builder=feature_builder,
    )

    ens_result = ensemble.predict(home_team, away_team)
    component_models = ens_result["component_models"]
    ensemble_probs = ens_result["ensemble_probabilities"]

    # Build feature vector for completeness scoring
    try:
        feature_vector = feature_builder.build_features_for_match(home_team, away_team)
    except Exception as exc:
        logger.warning(f"Could not build feature vector: {exc}")
        feature_vector = {}

    calibration_ece = _load_ece_from_benchmark()
    scorer = ConfidenceScorer(matches_df=matches_df, calibration_ece=calibration_ece)
    conf = scorer.score(
        home_team=home_team,
        away_team=away_team,
        component_models=component_models,
        ensemble_probs=ensemble_probs,
        feature_vector=feature_vector,
    )

    # Display results
    band_color = {"High": "green", "Medium": "yellow", "Low": "red"}.get(
        conf["confidence_band"], "white"
    )
    console.print()
    console.print(
        Panel(
            f"[bold]{home_team}[/bold] vs [bold]{away_team}[/bold]\n"
            f"Confidence: [{band_color}]{conf['confidence_score']:.0%}[/{band_color}] "
            f"([{band_color}]{conf['confidence_band']}[/{band_color}])",
            title="Prediction Confidence",
            style="blue",
        )
    )

    table = Table(title="Factor Breakdown", show_header=True, header_style="bold cyan")
    table.add_column("Factor", style="white")
    table.add_column("Score", style="green", justify="right")
    table.add_column("Weight", style="dim", justify="right")

    weights = {
        "model_agreement": 0.35,
        "calibration_quality": 0.25,
        "feature_completeness": 0.20,
        "historical_reliability": 0.15,
        "prediction_volatility": 0.05,
    }
    for factor, value in conf["factor_breakdown"].items():
        table.add_row(factor, f"{value:.4f}", f"{weights.get(factor, 0):.2f}")

    console.print()
    console.print(table)
    console.print()


def cmd_drift_check() -> None:
    """Check for prediction drift on the historical test set."""
    model_path = settings.outputs_dir / "models" / "model.pkl"
    if not model_path.exists():
        console.print(
            "[bold red]No trained model found.[/bold red] "
            "Run [bold]python main.py train[/bold] first."
        )
        return

    data_path = settings.data_dir / "processed" / "matches.csv"
    if not data_path.exists():
        data_path = settings.data_dir / "sample" / "matches.csv"
    matches_df = load_matches(data_path)

    elo = EloModel(config=settings.elo)
    elo.train_on_matches(matches_df)
    poisson = PoissonModel(config=settings.poisson)
    poisson.fit(matches_df)
    xgb_model = XGBoostMatchModel.load(model_path)
    feature_builder = FeatureBuilder(
        matches_df=matches_df,
        elo_model=elo,
        poisson_model=poisson,
    )
    ensemble = EnsembleEngine(
        elo_model=elo,
        poisson_model=poisson,
        xgb_model=xgb_model,
        feature_builder=feature_builder,
    )

    from src.diagnostics.drift_detection import DriftDetector
    detector = DriftDetector()

    test_df = matches_df[matches_df["date"] >= settings.ml.test_split_date].copy()

    console.print(Panel(
        f"Running drift check on [bold]{len(test_df)}[/bold] test matches…",
        style="blue",
    ))

    alert_count = 0
    for _, row in test_df.iterrows():
        try:
            pred = ensemble.predict(row["home_team"], row["away_team"])
            ep = pred["ensemble_probabilities"]
            probs = {
                "home_win": ep.get("home_win", 1 / 3),
                "draw": ep.get("draw", 1 / 3),
                "away_win": ep.get("away_win", 1 / 3),
            }
            alerts = detector.add_record(y_true=row["result"], probs=probs)
            alert_count += len(alerts)
        except Exception as exc:
            logger.debug(f"Skipping {row['home_team']} vs {row['away_team']}: {exc}")

    status = detector.status()

    table = Table(title="Drift Check Status", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="white")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Total records", str(status["total_records"]))
    table.add_row("Window size", str(status["window_size"]))
    table.add_row("Has baseline", "Yes" if status["has_baseline"] else "No")
    table.add_row("Alerts fired", str(alert_count))

    console.print()
    console.print(table)

    if alert_count > 0:
        console.print(f"\n[bold yellow]⚠ {alert_count} drift alert(s) detected[/bold yellow]\n")
    else:
        console.print("\n[bold green]✓ No drift detected[/bold green]\n")
