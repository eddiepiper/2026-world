from __future__ import annotations

import sys
from datetime import date

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from src.config.settings import settings
from src.ingestion.match_loader import load_matches
from src.models.elo_model import EloModel
from src.models.poisson_model import PoissonModel
from src.reports.markdown_reporter import MarkdownReporter
from src.reports.forecast_summary import ReportSection, generate_report

console = Console()


def _load_matches_df() -> pd.DataFrame:
    data_path = settings.data_dir / "processed" / "matches.csv"
    if not data_path.exists():
        data_path = settings.data_dir / "sample" / "matches.csv"
    return load_matches(data_path)


def cmd_report() -> None:
    """Generate a full forecast reliability report to outputs/reports/."""
    from src.diagnostics.confidence_scorer import ConfidenceScorer
    from src.diagnostics.drift_detection import DriftDetector
    from src.ensemble.ensemble_engine import EnsembleEngine, EnsembleWeights
    from src.features.feature_builder import FeatureBuilder
    from src.ml.xgboost_model import XGBoostMatchModel

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

    weight_path = settings.outputs_dir / "models" / "best_weight_config.json"
    if weight_path.exists():
        import json
        cfg = json.loads(weight_path.read_text())
        weights = EnsembleWeights(
            elo=cfg["elo"], poisson=cfg["poisson"], xgboost=cfg["xgboost"]
        )
    else:
        weights = EnsembleWeights()

    ensemble = EnsembleEngine(
        elo_model=elo,
        poisson_model=poisson,
        xgb_model=xgb_model,
        feature_builder=feature_builder,
        weights=weights,
    )

    console.print(Panel("Generating forecast reliability report…", style="blue"))

    reporter = MarkdownReporter()
    sections: list[ReportSection] = []

    # --- Benchmark section ---
    benchmark_csv = settings.outputs_dir / "evaluation" / "benchmark_results.csv"
    bm_md = reporter.benchmark_section(benchmark_csv)
    sections.append(ReportSection(title="Model Benchmark", content=bm_md))

    # --- ECE for confidence scorer ---
    ece = 0.05  # fallback
    if benchmark_csv.exists():
        try:
            df_bm = pd.read_csv(benchmark_csv)
            row = df_bm[df_bm["model"].str.contains("Ensemble", case=False, na=False)]
            if not row.empty and "ece" in row.columns:
                ece = float(row["ece"].iloc[0])
        except Exception:
            pass

    conf_scorer = ConfidenceScorer(matches_df=matches_df, calibration_ece=ece)
    detector = DriftDetector()

    test_matches = matches_df[
        pd.to_datetime(matches_df["date"]) >= pd.Timestamp(settings.ml.test_split_date)
    ].sort_values("date").head(200)

    sample_confidence = None
    for _, row in test_matches.iterrows():
        home, away = str(row["home_team"]), str(row["away_team"])
        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        outcome = "H" if hg > ag else ("D" if hg == ag else "A")
        try:
            result = ensemble.predict(home, away, pd.Timestamp(row["date"]))
            probs = result["ensemble_probabilities"]
            detector.add_record(outcome, probs)
            if sample_confidence is None:
                features = feature_builder.build_features_for_match(
                    home, away, pd.Timestamp(row["date"])
                )
                sample_confidence = conf_scorer.score(
                    home, away,
                    result["component_models"],
                    probs,
                    features,
                )
        except Exception as exc:
            logger.warning(f"Skipping {home} vs {away}: {exc}")

    if sample_confidence:
        sections.append(
            ReportSection(
                title="Confidence Assessment (Sample)",
                content=reporter.confidence_section(sample_confidence),
            )
        )

    drift_status = detector.status()
    sections.append(
        ReportSection(title="Drift Detection", content=reporter.drift_section(drift_status))
    )

    # --- SHAP section (optional) ---
    from src.explainability.shap_engine import _SHAP_AVAILABLE
    if _SHAP_AVAILABLE:
        try:
            from src.explainability.shap_engine import SHAPEngine
            from src.features.feature_builder import FEATURE_COLUMNS
            shap_engine = SHAPEngine(xgb_model=xgb_model)
            X_test_rows = []
            for _, row in test_matches.head(50).iterrows():
                try:
                    feat = feature_builder.build_features_for_match(
                        str(row["home_team"]), str(row["away_team"]),
                        pd.Timestamp(row["date"])
                    )
                    X_test_rows.append(feat)
                except Exception:
                    pass
            if X_test_rows:
                X_test = pd.DataFrame(X_test_rows)[FEATURE_COLUMNS]
                global_shap = shap_engine.global_shap(X_test)
                sections.append(
                    ReportSection(
                        title="Feature Importance (SHAP)",
                        content=reporter.shap_section(global_shap),
                    )
                )
        except Exception as exc:
            logger.warning(f"SHAP section skipped: {exc}")
            sections.append(
                ReportSection(title="Feature Importance (SHAP)", content="_SHAP unavailable._")
            )
    else:
        sections.append(
            ReportSection(title="Feature Importance (SHAP)", content="_Install shap for this section._")
        )

    out_dir = settings.outputs_dir / "reports"
    out_path = generate_report(sections, output_dir=out_dir)

    console.print(f"\n[green]Report written to:[/green] {out_path}\n")
