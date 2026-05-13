from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from loguru import logger


@dataclass
class ReportSection:
    title: str
    content: str


def generate_report(
    sections: list[ReportSection],
    output_dir: Path,
    report_date: date | None = None,
) -> Path:
    """Render sections into a dated markdown file.

    Returns the path of the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_date = report_date or date.today()
    filename = f"forecast_report_{report_date.isoformat()}.md"
    out_path = output_dir / filename

    header = (
        f"# Forecast Reliability Report\n\n"
        f"**Generated:** {report_date.isoformat()}\n\n---\n\n"
    )

    body_parts = [f"## {s.title}\n\n{s.content}\n" for s in sections]
    full_text = header + "\n".join(body_parts)

    out_path.write_text(full_text)
    logger.info(f"Forecast report written to {out_path}")
    return out_path
