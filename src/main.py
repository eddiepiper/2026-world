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


def main() -> None:
    setup_logging(settings.outputs_dir / "logs")
    args = sys.argv[1:]

    if not args:
        console.print("[bold red]Usage:[/bold red]")
        console.print("  python main.py predict <home_team> <away_team>")
        console.print("  python main.py simulate [team1 team2 ...]")
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
    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
