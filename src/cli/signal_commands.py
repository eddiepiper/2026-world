from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config.settings import settings
from src.signals.article_parser import filter_relevant, parse_articles
from src.signals.rss_collector import collect_all, load_raw
from src.signals.signal_classifier import classify_all, load_classified, save_classified
from src.signals.signal_extractor import extract_all
from src.signals.signal_reporter import render_full_report, save_classified_json, save_report
from src.signals.signal_review import BaseForecast, build_review
from src.signals.team_matcher import match_signal_to_teams

console = Console()

_SIGNALS_DIR = settings.outputs_dir / "signals"
_RAW_DIR = _SIGNALS_DIR / "raw"
_PROCESSED_DIR = _SIGNALS_DIR / "processed"
_REPORTS_DIR = _SIGNALS_DIR / "reports"


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def cmd_signals_collect() -> None:
    """Fetch RSS feeds and save raw articles."""
    console.print(Panel("Collecting World Cup signals from RSS feeds…", style="cyan"))
    articles = collect_all(output_dir=_SIGNALS_DIR)
    console.print(f"[green]Collected {len(articles)} articles.[/green]")

    relevant = filter_relevant(parse_articles(articles))
    console.print(f"[dim]{len(relevant)} articles matched football signal keywords.[/dim]")

    if not articles:
        console.print(
            "[yellow]No articles collected — check source URLs or network access.[/yellow]"
        )


def cmd_signals_classify() -> None:
    """Classify raw signals from today's collected articles."""
    console.print(Panel("Classifying signals…", style="cyan"))

    raw = load_raw(_RAW_DIR)
    if not raw:
        console.print(
            "[yellow]No raw articles for today. Run [bold]signals collect[/bold] first.[/yellow]"
        )
        return

    articles = parse_articles(raw)
    relevant = filter_relevant(articles)
    console.print(f"[dim]{len(relevant)} relevant articles to process[/dim]")

    raw_signals = extract_all(relevant)
    classified = classify_all(raw_signals)
    out_path = save_classified(classified, _PROCESSED_DIR)

    table = Table(
        title=f"Classified Signals — {_today()}",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Type", style="white")
    table.add_column("Severity", style="yellow")
    table.add_column("Phrase", style="dim")
    table.add_column("Source", style="dim")

    severity_order = {"critical": 0, "significant": 1, "minor": 2}
    sorted_signals = sorted(classified, key=lambda s: severity_order.get(s.severity, 9))

    for s in sorted_signals[:20]:
        sev_style = "bold red" if s.severity == "critical" else (
            "yellow" if s.severity == "significant" else "green"
        )
        table.add_row(
            s.signal_type,
            f"[{sev_style}]{s.severity}[/{sev_style}]",
            s.matched_phrase[:40],
            s.source_name[:30],
        )

    console.print()
    console.print(table)
    console.print(f"\n[dim]Saved to {out_path}[/dim]\n")


def cmd_signals_review(home_team: str = "Mexico", away_team: str = "South Africa") -> None:
    """Build a signal-adjusted review for a specific match."""
    console.print(
        Panel(
            f"Building signal review: [bold]{home_team}[/bold] vs [bold]{away_team}[/bold]",
            style="cyan",
        )
    )

    # Load base forecast from ensemble if available, otherwise use defaults
    base = _load_base_forecast(home_team, away_team)

    # Load today's classified signals
    signals = load_classified(_PROCESSED_DIR)
    if not signals:
        console.print(
            "[yellow]No classified signals found. "
            "Run [bold]signals classify[/bold] first.[/yellow]"
        )

    review = build_review(base, signals)

    # Display
    _print_review_table(review)

    console.print(
        f"\n[bold yellow]⚠  {review.disclaimer}[/bold yellow]\n"
    )

    out_path = save_report(review, _REPORTS_DIR)
    console.print(f"[dim]Report saved to {out_path}[/dim]\n")


def cmd_signals_report() -> None:
    """Generate signal summary report for today across all collected signals."""
    console.print(Panel("Generating daily signal report…", style="cyan"))

    signals = load_classified(_PROCESSED_DIR)
    if not signals:
        console.print("[yellow]No classified signals to report.[/yellow]")
        return

    out_path = save_classified_json(signals, _REPORTS_DIR)

    table = Table(title="Daily Signal Summary", show_header=True, header_style="bold cyan")
    table.add_column("Type", style="white")
    table.add_column("Count", justify="right", style="green")

    from collections import Counter
    counts = Counter(s.signal_type for s in signals)
    for sig_type, count in sorted(counts.items(), key=lambda x: -x[1]):
        table.add_row(sig_type, str(count))

    console.print()
    console.print(table)
    console.print(f"\n[dim]JSON report saved to {out_path}[/dim]\n")


def cmd_signals_run_daily(home_team: str = "Mexico", away_team: str = "South Africa") -> None:
    """Full daily pipeline: collect → classify → review → report."""
    console.print(
        Panel(
            f"Running full daily signal pipeline for [bold]{home_team}[/bold] "
            f"vs [bold]{away_team}[/bold]",
            style="bold cyan",
        )
    )
    cmd_signals_collect()
    cmd_signals_classify()
    cmd_signals_review(home_team, away_team)
    cmd_signals_report()
    console.print(
        Panel(
            "[green]Daily signal pipeline complete.[/green]\n"
            f"[dim]Outputs in {_SIGNALS_DIR}[/dim]",
            style="green",
        )
    )


# ── Helpers ─────────────────────────────────────────────────────────────────

def _load_base_forecast(home_team: str, away_team: str) -> BaseForecast:
    """Load ensemble forecast if model exists, else return illustrative defaults."""
    model_path = settings.outputs_dir / "models" / "model.pkl"
    if model_path.exists():
        try:
            from src.ml.xgboost_model import XGBoostMatchModel
            from src.features.feature_builder import FeatureBuilder
            from src.ensemble.ensemble_engine import EnsembleEngine
            from src.models.elo_model import EloModel
            from src.models.poisson_model import PoissonModel
            from src.ingestion.match_loader import load_matches

            data_path = settings.data_dir / "processed" / "matches.csv"
            if not data_path.exists():
                data_path = settings.data_dir / "sample" / "matches.csv"

            matches_df = load_matches(data_path)
            elo = EloModel(config=settings.elo)
            elo.train_on_matches(matches_df)
            poisson = PoissonModel(config=settings.poisson)
            poisson.fit(matches_df)
            xgb = XGBoostMatchModel.load(model_path)
            fb = FeatureBuilder(matches_df=matches_df, elo_model=elo, poisson_model=poisson)
            engine = EnsembleEngine(
                elo_model=elo,
                poisson_model=poisson,
                xgb_model=xgb,
                feature_builder=fb,
            )
            result = engine.predict(home_team, away_team)
            ep = result["ensemble_probabilities"]
            return BaseForecast(
                home_team=home_team,
                away_team=away_team,
                home_win_prob=ep["home_win"],
                draw_prob=ep["draw"],
                away_win_prob=ep["away_win"],
            )
        except Exception as exc:
            logger.warning(f"Ensemble unavailable ({exc}), using illustrative defaults")

    # Illustrative defaults for demo (Mexico vs South Africa)
    _defaults: dict[tuple[str, str], tuple[float, float, float]] = {
        ("mexico", "south africa"): (0.56, 0.25, 0.19),
    }
    key = (home_team.lower(), away_team.lower())
    h, d, a = _defaults.get(key, (0.40, 0.30, 0.30))
    logger.info(f"Using illustrative base forecast for {home_team} vs {away_team}")
    return BaseForecast(
        home_team=home_team,
        away_team=away_team,
        home_win_prob=h,
        draw_prob=d,
        away_win_prob=a,
    )


def _print_review_table(review) -> None:
    b = review.base
    table = Table(
        title=f"Signal Review — {b.home_team} vs {b.away_team}",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Outcome", style="white")
    table.add_column("Base Forecast", style="blue", justify="right")
    table.add_column("Signal-Adjusted", style="magenta", justify="right")
    table.add_column("Delta", style="yellow", justify="right")

    def _delta_str(base: float, adj: float) -> str:
        diff = adj - base
        sign = "+" if diff >= 0 else ""
        return f"{sign}{diff:.0%}"

    table.add_row(
        f"{b.home_team} Win",
        f"{b.home_win_prob:.0%}",
        f"{review.adjusted_home_win:.0%}",
        _delta_str(b.home_win_prob, review.adjusted_home_win),
    )
    table.add_row(
        "Draw",
        f"{b.draw_prob:.0%}",
        f"{review.adjusted_draw:.0%}",
        _delta_str(b.draw_prob, review.adjusted_draw),
    )
    table.add_row(
        f"{b.away_team} Win",
        f"{b.away_win_prob:.0%}",
        f"{review.adjusted_away_win:.0%}",
        _delta_str(b.away_win_prob, review.adjusted_away_win),
    )

    console.print()
    console.print(table)

    if review.signals_applied:
        console.print(f"\n[bold]Signals applied ({len(review.signals_applied)}):[/bold]")
        for s in review.signals_applied[:5]:
            sev_color = "red" if s.severity == "critical" else (
                "yellow" if s.severity == "significant" else "green"
            )
            console.print(
                f"  [{sev_color}][{s.severity.upper()}][/{sev_color}] "
                f"{s.signal_type}: {s.matched_phrase[:50]}"
            )
    else:
        console.print("\n[dim]No relevant signals found for this match.[/dim]")
