from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config.settings import settings
from src.ingestion.match_loader import load_matches
from src.ingestion.team_loader import load_teams
from src.models.elo_model import EloModel
from src.models.poisson_model import PoissonModel
from src.models.prediction_engine import PredictionEngine
from src.simulation.monte_carlo import MonteCarloSimulator
from src.utils.logger import setup_logging

console = Console()


def _build_engine() -> PredictionEngine:
    sample = settings.data_dir / "sample"
    matches_df = load_matches(sample / "matches.csv")
    elo = EloModel(config=settings.elo)
    elo.train_on_matches(matches_df)
    poisson = PoissonModel(config=settings.poisson)
    poisson.fit(matches_df)
    return PredictionEngine(
        elo_model=elo,
        poisson_model=poisson,
        elo_weight=settings.weights.elo,
    )


def _load_matches_df():
    """Load matches: prefer processed CSV, fall back to sample."""
    data_path = settings.data_dir / "processed" / "matches.csv"
    if not data_path.exists():
        data_path = settings.data_dir / "sample" / "matches.csv"
    return load_matches(data_path)


def cmd_predict(home_team: str, away_team: str) -> None:
    engine = _build_engine()
    result = engine.predict(home_team, away_team)

    logger.info(f"Prediction requested: {home_team} vs {away_team}")

    out_dir = settings.outputs_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    table = Table(
        title=f"[bold]{home_team}[/bold] vs [bold]{away_team}[/bold]",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Outcome", style="white")
    table.add_column("Probability", style="green", justify="right")

    table.add_row(f"{home_team} Win", f"{result['win_probability']:.1%}")
    table.add_row("Draw", f"{result['draw_probability']:.1%}")
    table.add_row(f"{away_team} Win", f"{result['loss_probability']:.1%}")
    table.add_section()
    table.add_row("Expected Home Goals", f"{result['expected_home_goals']:.2f}")
    table.add_row("Expected Away Goals", f"{result['expected_away_goals']:.2f}")

    console.print()
    console.print(table)

    # Phase 2: show ensemble predictions if a trained model exists
    model_path = settings.outputs_dir / "models" / "model.pkl"
    if model_path.exists():
        try:
            from src.ml.xgboost_model import XGBoostMatchModel
            from src.features.feature_builder import FeatureBuilder
            from src.ensemble.ensemble_engine import EnsembleEngine

            matches_df = _load_matches_df()
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
            ep = ens_result["ensemble_probabilities"]

            ens_table = Table(
                title=f"[bold]Phase 2 Ensemble[/bold] — {home_team} vs {away_team}",
                show_header=True,
                header_style="bold magenta",
            )
            ens_table.add_column("Outcome", style="white")
            ens_table.add_column("Ensemble", style="magenta", justify="right")
            ens_table.add_column("Elo", style="dim", justify="right")
            ens_table.add_column("Poisson", style="dim", justify="right")
            ens_table.add_column("XGBoost", style="dim", justify="right")

            comp = ens_result["component_models"]
            ens_table.add_row(
                f"{home_team} Win",
                f"{ep['home_win']:.1%}",
                f"{comp['elo']['home_win']:.1%}",
                f"{comp['poisson']['home_win']:.1%}",
                f"{comp['xgboost']['home_win']:.1%}",
            )
            ens_table.add_row(
                "Draw",
                f"{ep['draw']:.1%}",
                f"{comp['elo']['draw']:.1%}",
                f"{comp['poisson']['draw']:.1%}",
                f"{comp['xgboost']['draw']:.1%}",
            )
            ens_table.add_row(
                f"{away_team} Win",
                f"{ep['away_win']:.1%}",
                f"{comp['elo']['away_win']:.1%}",
                f"{comp['poisson']['away_win']:.1%}",
                f"{comp['xgboost']['away_win']:.1%}",
            )
            ens_table.add_section()
            ens_table.add_row(
                "Confidence",
                f"{ens_result['confidence_score']:.1%}",
                "", "", "",
            )

            console.print()
            console.print(ens_table)
        except Exception as exc:
            logger.warning(f"Ensemble prediction failed: {exc}")
            console.print(f"[dim]Ensemble unavailable: {exc}[/dim]")
    else:
        console.print(
            "[dim]No ensemble model — run [bold]python main.py train[/bold] first[/dim]"
        )

    console.print()


def cmd_simulate(teams: list[str] | None = None) -> None:
    engine = _build_engine()

    if teams is None:
        sample = settings.data_dir / "sample"
        teams_df = load_teams(sample / "teams.csv")
        teams = teams_df["team_name"].tolist()[:16]

    n = settings.simulation.n_simulations
    console.print(
        Panel(
            f"Running [bold]{n:,}[/bold] simulations for [bold]{len(teams)}[/bold] teams…",
            style="blue",
        )
    )

    simulator = MonteCarloSimulator(
        engine=engine,
        n_simulations=n,
        random_seed=settings.simulation.random_seed,
    )
    df = simulator.run(teams)

    out_dir = settings.outputs_dir / "simulations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tournament_simulation.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Simulation results saved to {out_path}")

    table = Table(
        title="World Cup 2026 — Tournament Simulation",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Team", style="bold white")
    table.add_column("Win %", style="green", justify="right")
    table.add_column("Final %", style="yellow", justify="right")
    table.add_column("Semi %", style="blue", justify="right")

    for i, row in df.head(16).iterrows():
        table.add_row(
            str(i + 1),
            row["team"],
            f"{row['winner_prob']:.1%}",
            f"{row['finalist_prob']:.1%}",
            f"{row['semifinal_prob']:.1%}",
        )

    console.print()
    console.print(table)
    console.print(f"\n[dim]Full results saved to {out_path}[/dim]\n")


def cmd_train() -> None:
    """Train XGBoost model and save artifacts to outputs/models/."""
    from src.ml.trainer import ModelTrainer

    matches_df = _load_matches_df()
    elo = EloModel(config=settings.elo)
    elo.train_on_matches(matches_df)
    poisson = PoissonModel(config=settings.poisson)
    poisson.fit(matches_df)

    trainer = ModelTrainer(
        matches_df=matches_df,
        elo_model=elo,
        poisson_model=poisson,
    )
    output_dir = settings.outputs_dir / "models"

    console.print(Panel("Training XGBoost model…", style="blue"))
    metrics = trainer.run(output_dir)

    # Also capture train/test sizes from the split
    feature_df = trainer.build_feature_matrix()
    train_df, test_df = trainer.split(feature_df)

    table = Table(
        title="Training Results",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="white")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Train size", str(len(train_df)))
    table.add_row("Test size", str(len(test_df)))
    table.add_row("Accuracy", f"{metrics['accuracy']:.4f}")
    table.add_row("Log Loss", f"{metrics['log_loss']:.4f}")
    table.add_row("Brier Score", f"{metrics['brier_score']:.4f}")

    console.print()
    console.print(table)
    console.print(
        f"\n[dim]Model artifacts saved to {output_dir}[/dim]\n"
    )


def _cmd_evaluate_or_benchmark(detailed: bool = False) -> None:
    """Shared implementation for evaluate and benchmark commands."""
    from src.ml.xgboost_model import XGBoostMatchModel
    from src.features.feature_builder import FeatureBuilder
    from src.evaluation.benchmark import ModelBenchmarker

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
        matches_df=matches_df,
        elo_model=elo,
        poisson_model=poisson,
    )

    benchmarker = ModelBenchmarker(
        matches_df=matches_df,
        elo_model=elo,
        poisson_model=poisson,
        xgb_model=xgb_model,
        feature_builder=feature_builder,
        test_split_date=settings.ml.test_split_date,
    )

    console.print(Panel("Running model benchmark…", style="blue"))
    results = benchmarker.run()

    if not results:
        console.print("[yellow]No test data available for benchmarking.[/yellow]")
        return

    # Save outputs
    out_dir = settings.outputs_dir / "evaluation"
    benchmarker.save(results, out_dir)

    # Build Rich table
    title = "Model Benchmark" if not detailed else "Detailed Model Benchmark"
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Model", style="white", min_width=24)
    table.add_column("Accuracy", style="green", justify="right")
    table.add_column("Log Loss", style="yellow", justify="right")
    table.add_column("Brier Score", style="blue", justify="right")

    if detailed:
        best_name = benchmarker.best_model(results)
    else:
        best_name = None

    for r in results:
        name = r.model_name
        m = r.metrics
        winner_marker = " [bold green]<-- BEST[/bold green]" if (detailed and name == best_name) else ""
        table.add_row(
            name + winner_marker,
            f"{m['accuracy']:.4f}",
            f"{m['log_loss']:.4f}",
            f"{m['brier_score']:.4f}",
        )

    console.print()
    console.print(table)

    if detailed and best_name:
        console.print(
            f"\n[bold green]Winner:[/bold green] {best_name} (lowest log loss)\n"
        )

    console.print(f"\n[dim]Results saved to {out_dir}[/dim]\n")


def cmd_evaluate() -> None:
    """Evaluate trained model — prints benchmark results table."""
    _cmd_evaluate_or_benchmark(detailed=False)


def cmd_benchmark() -> None:
    """Detailed model comparison — highlights the best performing model."""
    _cmd_evaluate_or_benchmark(detailed=True)


def main() -> None:
    setup_logging(settings.outputs_dir / "logs")
    args = sys.argv[1:]

    if not args:
        console.print("[bold red]Usage:[/bold red]")
        console.print("  python main.py predict <home_team> <away_team>")
        console.print("  python main.py simulate [team1 team2 ...]")
        console.print("  python main.py train")
        console.print("  python main.py evaluate")
        console.print("  python main.py benchmark")
        console.print("  python main.py optimize")
        sys.exit(1)

    command = args[0]

    if command == "predict":
        if len(args) != 3:
            console.print("[red]predict requires exactly two team names[/red]")
            sys.exit(1)
        cmd_predict(args[1], args[2])
    elif command == "simulate":
        extra = args[1:] if len(args) > 1 else None
        cmd_simulate(extra)
    elif command == "train":
        cmd_train()
    elif command == "evaluate":
        cmd_evaluate()
    elif command == "benchmark":
        cmd_benchmark()
    elif command == "optimize":
        from src.cli.optimize_cmd import cmd_optimize
        cmd_optimize()
    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
