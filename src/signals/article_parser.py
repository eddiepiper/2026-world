from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# URL path substrings that indicate a non-football sport
_NON_FOOTBALL_URL_KEYWORDS: frozenset[str] = frozenset([
    "rugby", "cricket", "tennis", "nfl", "formula-1", "formula1",
    "golf", "basketball", "nba", "nhl", "swimming", "cycling",
    "boxing", "american-football",
])

# Compiled regex for non-football words in article titles
_NON_FOOTBALL_TITLE_RE = re.compile(
    r"\b(?:rugby|cricket|tennis|wimbledon|nfl|quarterback|golf|basketball|nba|f1|swimming|baseball)\b"
    r"|formula\s+1|formula\s+one|grand\s+prix|super\s+bowl|rugby\s+union|rugby\s+league",
    re.IGNORECASE,
)


@dataclass
class ParsedArticle:
    source_name: str
    title: str
    body: str
    link: str
    published_raw: str
    collected_at: str
    source_tags: list[str]

    @property
    def full_text(self) -> str:
        return f"{self.title}. {self.body}"

    @property
    def word_count(self) -> int:
        return len(self.full_text.split())


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r" {2,}", " ", text).strip()


def parse_article(raw: dict[str, Any]) -> ParsedArticle:
    title = _strip_html(raw.get("title", ""))
    summary = _strip_html(raw.get("summary", ""))
    return ParsedArticle(
        source_name=raw.get("source_name", ""),
        title=title,
        body=summary,
        link=raw.get("link", ""),
        published_raw=raw.get("published", ""),
        collected_at=raw.get("collected_at", ""),
        source_tags=raw.get("source_tags", []),
    )


def parse_articles(raw_list: list[dict[str, Any]]) -> list[ParsedArticle]:
    return [parse_article(r) for r in raw_list]


def is_football_article(article: ParsedArticle) -> bool:
    """Return False if the article is clearly about a non-football sport."""
    link_lower = article.link.lower()
    if any(kw in link_lower for kw in _NON_FOOTBALL_URL_KEYWORDS):
        return False
    if _NON_FOOTBALL_TITLE_RE.search(article.full_text):
        return False
    return True


def filter_relevant(
    articles: list[ParsedArticle],
    keywords: list[str] | None = None,
) -> list[ParsedArticle]:
    """Keep only football articles containing at least one signal keyword."""
    if keywords is None:
        keywords = [
            "injury", "injured", "doubt", "ruled out", "miss",
            "suspension", "suspended", "ban", "yellow card", "red card",
            "lineup", "squad", "starter", "benched", "fitness",
            "world cup", "qualifier", "international football",
            "form", "win streak", "loss streak",
        ]
    kw_lower = [k.lower() for k in keywords]

    def _matches(article: ParsedArticle) -> bool:
        text = article.full_text.lower()
        return any(k in text for k in kw_lower)

    return [a for a in articles if is_football_article(a) and _matches(a)]
