from __future__ import annotations

import sys

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config.settings import settings
from src.ingestion.match_loader import load_matches
from src.models.elo_model import EloModel
from src.models.poisson_model import PoissonModel

console = Console()


def _load_matches_df() -> pd.DataFrame:
    data_path = settings.data_dir / "processed" / "matches.csv"
    if not data_path.exists():
        data_path = settings.data_dir / "sample" / "matches.csv"
    return load_matches(data_path)


def cmd_scenarios(home_team: str = "Brazil", away_team: str = "France") -> None:
    """Run scenario analysis + upset check for the given match."""
    from src.features.feature_builder import FeatureBuilder
    from src.ml.xgboost_model import XGBoostMatchModel
    from src.scenarios.scenario_runner import ScenarioRunner, Perturbation
    from src.scenarios.upset_analysis import UpsetAnalyzer

    model_path = settings.outputs_dir / "models" / "model.pkl"
    if not model_path.exists():
        console.print(
            "[bold red]No trained model found.[/bold red] "
            "Run [bold]python main.py train[/bold] first."
        )
        sys.exit(1)

    xgb_model = XGBoostMatchModel.load(model_path)
    matches_df = _load_matches_df()

    elo = EloModel(config=settings.elo)
    elo.train_on_matches(matches_df)
    poisson = PoissonModel(config=settings.poisson)
    poisson.fit(matches_df)

    feature_builder = FeatureBuilder(
        matches_df=matches_df, elo_model=elo, poisson_model=poisson
    )

    # Load optimized weights if available, else fall back to settings
    weight_path = settings.outputs_dir / "models" / "best_weight_config.json"
    if weight_path.exists():
        import json
        cfg = json.loads(weight_path.read_text())
        weights = (cfg["elo"], cfg["poisson"], cfg["xgboost"])
    else:
        weights = (
            settings.ensemble.elo_weight,
            settings.ensemble.poisson_weight,
            settings.ensemble.xgboost_weight,
        )

    runner = ScenarioRunner(
        elo_model=elo,
        poisson_model=poisson,
        xgb_model=xgb_model,
        feature_builder=feature_builder,
        weights=weights,
    )

    console.print(Panel(f"Scenario analysis: {home_team} vs {away_team}", style="blue"))

    # Baseline
    baseline_result = runner.run(home_team, away_team, [])
    bp = baseline_result["baseline"]

    # Scenario 1: away team loses attack strength
    scenario1 = runner.run(
        home_team, away_team,
        [Perturbation(param="attack_strength", team=away_team, delta=-0.3)]
    )
    # Scenario 2: home team loses recent form
    scenario2 = runner.run(
        home_team, away_team,
        [Perturbation(param="recent_form", team=home_team, delta=-0.2)]
    )

    table = Table(
        title=f"Scenario Analysis — {home_team} vs {away_team}",
        show_header=True, header_style="bold cyan"
    )
    table.add_column("Scenario", style="white")
    table.add_column(f"{home_team} Win", style="green", justify="right")
    table.add_column("Draw", style="yellow", justify="right")
    table.add_column(f"{away_team} Win", style="red", justify="right")

    table.add_row("Baseline",
                  f"{bp['home_win']:.1%}", f"{bp['draw']:.1%}", f"{bp['away_win']:.1%}")
    p1 = scenario1["perturbed"]
    table.add_row(f"{away_team} −30% attack",
                  f"{p1['home_win']:.1%}", f"{p1['draw']:.1%}", f"{p1['away_win']:.1%}")
    p2 = scenario2["perturbed"]
    table.add_row(f"{home_team} −20% form",
                  f"{p2['home_win']:.1%}", f"{p2['draw']:.1%}", f"{p2['away_win']:.1%}")

    console.print()
    console.print(table)

    # Upset check
    analyzer = UpsetAnalyzer()
    flag = analyzer.check(
        home_team=home_team,
        away_team=away_team,
        home_win_prob=bp["home_win"],
        away_win_prob=bp["away_win"],
        confidence_band="Medium",
        stability_band="Stable",
    )

    if flag.is_upset:
        console.print("\n[bold yellow]Upset Alert:[/bold yellow]")
        for reason in flag.reasons:
            console.print(f"  • {reason}")
    else:
        console.print("\n[green]No upset potential flagged for baseline prediction.[/green]")

    console.print()
