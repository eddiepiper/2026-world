from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
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

# Phrases that indicate a match has already been played (post-match reports)
_POST_MATCH_RE = re.compile(
    r"\b(?:match report|full[\s-]time|full time|ft:|final score|highlights|"
    r"recap|as it happened|result:|player ratings|talking points)\b",
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


def _is_fresh(article: ParsedArticle, max_age_days: int) -> bool:
    """Return True if article is within max_age_days of now, or if date is unparseable."""
    if not article.published_raw:
        return True  # fail-open: no date → don't filter
    try:
        pub_dt = parsedate_to_datetime(article.published_raw)
        # Make timezone-aware if naive
        if pub_dt.tzinfo is None:
            pub_dt = pub_dt.replace(tzinfo=timezone.utc)
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        return pub_dt >= cutoff
    except Exception:
        return True  # fail-open: unparseable date → don't filter


def _is_post_match(article: ParsedArticle) -> bool:
    """Return True if the article is a post-match report."""
    return bool(_POST_MATCH_RE.search(article.full_text))


def filter_relevant(
    articles: list[ParsedArticle],
    keywords: list[str] | None = None,
    max_age_days: int = 7,
) -> list[ParsedArticle]:
    """Keep only fresh, pre-match football articles containing at least one signal keyword."""
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

    return [
        a for a in articles
        if is_football_article(a)
        and _is_fresh(a, max_age_days)
        and not _is_post_match(a)
        and _matches(a)
    ]
