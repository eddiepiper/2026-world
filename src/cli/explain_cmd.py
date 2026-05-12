from __future__ import annotations

import sys

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from src.config.settings import settings
from src.ingestion.match_loader import load_matches
from src.models.elo_model import EloModel
from src.models.poisson_model import PoissonModel

console = Console()


def cmd_explain(home_team: str, away_team: str, outcome: str = "H") -> None:
    """Print SHAP-based top drivers for the given match. Requires shap + trained model."""
    from src.explainability.shap_engine import _SHAP_AVAILABLE, _SHAP_UNAVAILABLE_MSG

    if not _SHAP_AVAILABLE:
        console.print(f"[bold yellow]{_SHAP_UNAVAILABLE_MSG}[/bold yellow]")
        sys.exit(0)

    model_path = settings.outputs_dir / "models" / "model.pkl"
    if not model_path.exists():
        console.print(
            "[bold red]No trained model found.[/bold red] "
            "Run [bold]python main.py train[/bold] first."
        )
        sys.exit(1)

    from src.explainability.shap_engine import SHAPEngine
    from src.explainability.prediction_explainer import PredictionExplainer
    from src.features.feature_builder import FeatureBuilder
    from src.ml.xgboost_model import XGBoostMatchModel

    xgb_model = XGBoostMatchModel.load(model_path)

    data_path = settings.data_dir / "processed" / "matches.csv"
    if not data_path.exists():
        data_path = settings.data_dir / "sample" / "matches.csv"
    matches_df = load_matches(data_path)

    elo = EloModel(config=settings.elo)
    elo.train_on_matches(matches_df)
    poisson = PoissonModel(config=settings.poisson)
    poisson.fit(matches_df)

    feature_builder = FeatureBuilder(
        matches_df=matches_df, elo_model=elo, poisson_model=poisson
    )

    match_date = pd.Timestamp.today()
    features = feature_builder.build_features_for_match(home_team, away_team, match_date)
    feature_row = pd.DataFrame([features])

    console.print(Panel(
        f"Computing SHAP explanation for {home_team} vs {away_team}…",
        style="blue"
    ))

    try:
        shap_engine = SHAPEngine(xgb_model=xgb_model)
        explainer = PredictionExplainer(shap_engine=shap_engine)
        explanation = explainer.explain(
            home_team=home_team,
            away_team=away_team,
            feature_row=feature_row,
            outcome=outcome,
            top_n=5,
        )

        outcome_label = {"H": "Home win", "D": "Draw", "A": "Away win"}.get(outcome, outcome)
        console.print(f"\n[bold]{home_team} vs {away_team}[/bold] — Top drivers ({outcome_label}):\n")

        for driver in explanation["top_drivers"]:
            sign = "+" if driver["shap_value"] >= 0 else ""
            color = "green" if driver["shap_value"] >= 0 else "red"
            console.print(
                f"  {driver['rank']}. {driver['feature']:<28} "
                f"[{color}]{sign}{driver['shap_value']:.4f}[/{color}]"
            )

        out_dir = settings.outputs_dir / "reports" / "explanations"
        explainer.save(explanation, out_dir, match_date.strftime("%Y-%m-%d"))
        console.print(f"\n[dim]Saved to {out_dir}[/dim]\n")

    except ImportError as exc:
        console.print(f"[bold yellow]{exc}[/bold yellow]")
        sys.exit(0)
    except Exception as exc:
        logger.error(f"SHAP explanation failed: {exc}")
        console.print(f"[red]Explanation failed: {exc}[/red]")
        sys.exit(1)
