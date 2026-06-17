from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SignalSource:
    name: str
    url: str
    category: str  # rss | api | scrape
    tags: list[str] = field(default_factory=list)
    enabled: bool = True


# Default World Cup / football news RSS sources
DEFAULT_SOURCES: list[SignalSource] = [
    SignalSource(
        name="BBC Sport Football",
        url="https://feeds.bbci.co.uk/sport/football/rss.xml",
        category="rss",
        tags=["general", "injuries", "lineup"],
    ),
    SignalSource(
        name="ESPN FC",
        url="https://www.espn.com/espn/rss/soccer/news",
        category="rss",
        tags=["general", "transfers"],
    ),
    SignalSource(
        name="Sky Sports Football",
        url="https://www.skysports.com/rss/12040",
        category="rss",
        tags=["injuries", "lineup", "form"],
    ),
    SignalSource(
        name="Goal.com",
        url="https://www.goal.com/feeds/en/news",
        category="rss",
        tags=["world cup", "international"],
    ),
    SignalSource(
        name="FIFA News",
        url="https://www.fifa.com/rss-feeds/news_en.xml",
        category="rss",
        tags=["world cup", "official"],
    ),
]


def get_enabled_sources() -> list[SignalSource]:
    return [s for s in DEFAULT_SOURCES if s.enabled]


def get_sources_by_tag(tag: str) -> list[SignalSource]:
    return [s for s in DEFAULT_SOURCES if s.enabled and tag in s.tags]
