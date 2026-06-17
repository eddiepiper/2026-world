from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from src.signals.source_config import SignalSource, get_enabled_sources

try:
    import feedparser  # type: ignore[import-untyped]
    _FEEDPARSER_AVAILABLE = True
except ImportError:
    _FEEDPARSER_AVAILABLE = False


def _require_feedparser() -> None:
    if not _FEEDPARSER_AVAILABLE:
        raise ImportError(
            "feedparser is required for RSS collection. "
            "Install it with: pip install feedparser"
        )


def _parse_feed(source: SignalSource) -> list[dict[str, Any]]:
    """Fetch and parse a single RSS feed. Returns list of entry dicts."""
    _require_feedparser()
    try:
        feed = feedparser.parse(source.url)
        entries = []
        for entry in feed.entries:
            entries.append({
                "source_name": source.name,
                "source_url": source.url,
                "source_tags": source.tags,
                "title": getattr(entry, "title", ""),
                "summary": getattr(entry, "summary", ""),
                "link": getattr(entry, "link", ""),
                "published": getattr(entry, "published", ""),
                "collected_at": datetime.now(timezone.utc).isoformat(),
            })
        logger.info(f"Collected {len(entries)} articles from {source.name}")
        return entries
    except Exception as exc:
        logger.warning(f"Failed to fetch {source.name} ({source.url}): {exc}")
        return []


def collect_all(
    sources: list[SignalSource] | None = None,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Collect articles from all enabled sources.
    Saves raw JSON to output_dir/raw/<date>.json.
    Returns the list of collected article dicts.
    """
    if sources is None:
        sources = get_enabled_sources()

    all_articles: list[dict[str, Any]] = []
    for source in sources:
        all_articles.extend(_parse_feed(source))

    if output_dir is not None:
        raw_dir = output_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out_path = raw_dir / f"signals_{date_str}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(all_articles, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(all_articles)} raw articles to {out_path}")

    return all_articles


def load_raw(raw_dir: Path, date_str: str | None = None) -> list[dict[str, Any]]:
    """Load previously collected raw articles from disk."""
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = raw_dir / f"signals_{date_str}.json"
    if not path.exists():
        logger.warning(f"No raw signals file found for {date_str} at {path}")
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
