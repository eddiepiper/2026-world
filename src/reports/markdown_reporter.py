from __future__ import annotations


class MarkdownReporter:
    """Render diagnostic signals and metadata into markdown sections."""

    def confidence_section(self, confidence: dict) -> str:
        score = confidence.get("confidence_score", 0.0)
        band = confidence.get("confidence_band", "Unknown")
        breakdown = confidence.get("factor_breakdown", {})

        lines = [
            "## Confidence Assessment\n",
            f"**Overall Score:** {score:.3f} | **Band:** {band}\n",
        ]
        if breakdown:
            lines.append("| Factor | Score |")
            lines.append("|--------|-------|")
            for factor, value in breakdown.items():
                lines.append(f"| {factor.replace('_', ' ').title()} | {value:.3f} |")

        return "\n".join(lines)

    def drift_section(self, drift: dict) -> str:
        status = drift.get("status", drift.get("has_baseline", False) and "ok" or "no_baseline")
        n = drift.get("total_records", drift.get("n_records", 0))
        alerts = drift.get("alerts", [])

        lines = [
            "## Drift Detection\n",
            f"**Status:** {str(status).upper()} | **Records analysed:** {n}\n",
        ]

        if alerts:
            lines.append("**Alerts:**")
            for alert in alerts:
                metric = alert.get("metric", "unknown")
                current = alert.get("current_value", 0.0)
                baseline = alert.get("baseline_mean", 0.0)
                severity = alert.get("severity", "warning")
                lines.append(
                    f"- [{severity.upper()}] {metric}: "
                    f"{current:.4f} (baseline {baseline:.4f})"
                )
        else:
            lines.append("_No drift alerts._")

        return "\n".join(lines)

    def stability_section(self, stability: dict) -> str:
        band = stability.get("stability_band", "Unknown")
        home = stability.get("home_team", "")
        away = stability.get("away_team", "")
        stds = stability.get("std_probabilities", {})

        lines = [
            "## Prediction Stability\n",
            f"**Match:** {home} vs {away}",
            f"**Stability Band:** {band}\n",
        ]
        if stds:
            lines.append("| Outcome | Std Dev |")
            lines.append("|---------|---------|")
            for outcome, std in stds.items():
                lines.append(f"| {outcome.replace('_', ' ').title()} | {std:.4f} |")

        return "\n".join(lines)

    def benchmark_section(self, benchmark_csv_path) -> str:
        from pathlib import Path
        try:
            import pandas as pd
            path = Path(benchmark_csv_path)
            if not path.exists():
                return "## Model Benchmark\n\n_Benchmark results not available._"
            df = pd.read_csv(path)
            # Find model name column and log_loss column
            name_col = next((c for c in df.columns if "model" in c.lower()), df.columns[0])
            ll_col = next((c for c in df.columns if "log_loss" in c.lower()), None)
            if ll_col is None:
                return "## Model Benchmark\n\n_log_loss column not found._"
            best_row = df.loc[df[ll_col].idxmin()]
            lines = [
                "## Model Benchmark\n",
                f"**Best Model:** {best_row[name_col]} "
                f"(log loss {best_row[ll_col]:.4f})\n",
                f"| {name_col.title()} | Accuracy | Log Loss | Brier |",
                "|-------|----------|----------|-------|",
            ]
            acc_col = next((c for c in df.columns if "accuracy" in c.lower()), None)
            brier_col = next((c for c in df.columns if "brier" in c.lower()), None)
            for _, row in df.iterrows():
                acc = f"{row[acc_col]:.4f}" if acc_col else "N/A"
                brier = f"{row[brier_col]:.4f}" if brier_col else "N/A"
                lines.append(f"| {row[name_col]} | {acc} | {row[ll_col]:.4f} | {brier} |")
            return "\n".join(lines)
        except Exception as exc:
            return f"## Model Benchmark\n\n_Unavailable: {exc}_"

    def shap_section(self, global_shap: dict | None, top_n: int = 10) -> str:
        if not global_shap:
            return "## Feature Importance (SHAP)\n\n_SHAP not available._"
        sorted_features = sorted(global_shap.items(), key=lambda x: x[1], reverse=True)
        lines = [
            "## Feature Importance (SHAP)\n",
            "| Rank | Feature | Mean |SHAP| |",
            "|------|---------|--------------|",
        ]
        for i, (name, val) in enumerate(sorted_features[:top_n], 1):
            lines.append(f"| {i} | {name} | {val:.6f} |")
        return "\n".join(lines)
