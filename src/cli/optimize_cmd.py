from __future__ import annotations

from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config.settings import settings
from src.features.feature_builder import FeatureBuilder
from src.ingestion.match_loader import load_matches
from src.ml.xgboost_model import XGBoostMatchModel
from src.models.elo_model import EloModel
from src.models.poisson_model import PoissonModel
from src.optimization.ensemble_optimizer import WeightOptimizer

console = Console()


def cmd_optimize() -> None:
    """Find optimal ensemble weights using SLSQP and save to outputs/models/."""
    model_path = settings.outputs_dir / "models" / "model.pkl"
    if not model_path.exists():
        console.print(
            "[bold red]No trained model found.[/bold red] "
            "Run [bold]python main.py train[/bold] first."
        )
        return

    # Load data
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

    console.print(Panel("Optimizing ensemble weights…", style="blue"))

    optimizer = WeightOptimizer(
        matches_df=matches_df,
        elo_model=elo,
        poisson_model=poisson,
        xgb_model=xgb_model,
        feature_builder=feature_builder,
    )
    result = optimizer.optimize()

    out_path = settings.outputs_dir / "models" / "best_weight_config.json"
    optimizer.save(out_path, result)

    table = Table(title="Optimized Ensemble Weights", show_header=True, header_style="bold cyan")
    table.add_column("Component", style="white")
    table.add_column("Weight", style="green", justify="right")

    table.add_row("Elo", f"{result['elo']:.4f}")
    table.add_row("Poisson", f"{result['poisson']:.4f}")
    table.add_row("XGBoost", f"{result['xgboost']:.4f}")
    table.add_section()
    table.add_row("Baseline Log Loss", f"{result['baseline_log_loss']:.4f}")
    table.add_row("Optimized Log Loss", f"{result['optimized_log_loss']:.4f}")
    table.add_row("Improvement", f"{result['improvement']:.4f}")

    console.print()
    console.print(table)
    console.print(f"\n[dim]Saved to {out_path}[/dim]\n")
