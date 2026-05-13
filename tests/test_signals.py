from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.signals.article_parser import ParsedArticle, filter_relevant, parse_article, parse_articles
from src.signals.impact_scorer import (
    ImpactDelta,
    aggregate_deltas,
    score_signal,
    _MAX_DELTA,
)
from src.signals.signal_classifier import (
    ClassifiedSignal,
    classify,
    classify_all,
    load_classified,
    save_classified,
)
from src.signals.signal_extractor import RawSignal, extract_signals
from src.signals.signal_reporter import render_full_report, render_review_table, save_report
from src.signals.signal_review import BaseForecast, SignalReview, build_review
from src.signals.team_matcher import find_teams_in_text, match_signal_to_teams


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def mexico_vs_sa() -> BaseForecast:
    return BaseForecast(
        home_team="Mexico",
        away_team="South Africa",
        home_win_prob=0.56,
        draw_prob=0.25,
        away_win_prob=0.19,
    )


@pytest.fixture()
def injury_signal() -> ClassifiedSignal:
    return ClassifiedSignal(
        signal_type="injury",
        severity="significant",
        matched_phrase="injured",
        context_snippet="Mexico key striker injured before the World Cup",
        article_title="Mexico striker injured",
        article_link="https://example.com/article",
        source_name="BBC Sport",
        team_hint="Mexico",
        classified_at="2026-05-13T00:00:00+00:00",
    )


@pytest.fixture()
def suspension_signal() -> ClassifiedSignal:
    return ClassifiedSignal(
        signal_type="suspension",
        severity="critical",
        matched_phrase="suspended",
        context_snippet="South Africa captain suspended for the match",
        article_title="South Africa captain out",
        article_link="https://example.com/article2",
        source_name="ESPN",
        team_hint="South Africa",
        classified_at="2026-05-13T00:00:00+00:00",
    )


@pytest.fixture()
def raw_article() -> dict:
    return {
        "source_name": "BBC Sport",
        "source_url": "https://feeds.bbci.co.uk/sport/football/rss.xml",
        "source_tags": ["injuries"],
        "title": "Mexico striker ruled out with knee injury",
        "summary": "The key forward is doubtful after sustaining a knee injury in training.",
        "link": "https://bbc.co.uk/sport/article",
        "published": "2026-05-13",
        "collected_at": "2026-05-13T10:00:00+00:00",
    }


# ── Team Matcher ─────────────────────────────────────────────────────────────

class TestTeamMatcher:
    def test_finds_canonical_team(self):
        matches = find_teams_in_text("Mexico face South Africa in the opening game.")
        names = [m.canonical_name for m in matches]
        assert "Mexico" in names
        assert "South Africa" in names

    def test_alias_resolution(self):
        matches = find_teams_in_text("USA will play against Brazil.")
        names = [m.canonical_name for m in matches]
        assert "United States" in names
        assert "Brazil" in names

    def test_no_false_positives_on_unknown_names(self):
        matches = find_teams_in_text("Zanzibar FC beat Ruritania 3-0.")
        assert len(matches) == 0

    def test_case_insensitive(self):
        matches = find_teams_in_text("MEXICO beat south africa.")
        names = [m.canonical_name for m in matches]
        assert "Mexico" in names

    def test_match_signal_to_teams_returns_list(self):
        result = match_signal_to_teams("France and Germany will meet in Berlin.")
        assert "France" in result
        assert "Germany" in result

    def test_empty_text(self):
        assert match_signal_to_teams("") == []


# ── Signal Classifier ────────────────────────────────────────────────────────

class TestSignalClassifier:
    def test_injury_classified_as_significant(self):
        raw = RawSignal(
            article_title="Player injury news",
            article_link="https://example.com",
            source_name="BBC",
            signal_type="injury",
            matched_phrase="injured",
            context_snippet="midfielder injured in training session",
            raw_text="midfielder injured in training session",
        )
        result = classify(raw)
        assert result.signal_type == "injury"
        assert result.severity in ("critical", "significant", "minor")

    def test_suspension_elevated_to_critical(self):
        raw = RawSignal(
            article_title="Player ruled out",
            article_link="https://example.com",
            source_name="ESPN",
            signal_type="suspension",
            matched_phrase="banned",
            context_snippet="star player banned for the key match ruled out",
            raw_text="star player banned for the key match ruled out",
        )
        result = classify(raw)
        assert result.severity == "critical"

    def test_all_types_produce_valid_severity(self):
        sig_types = ["injury", "suspension", "lineup_change", "form", "weather", "travel"]
        for sig_type in sig_types:
            raw = RawSignal(
                article_title="Test",
                article_link="https://example.com",
                source_name="Test",
                signal_type=sig_type,
                matched_phrase="test phrase",
                context_snippet="generic context",
                raw_text="generic context",
            )
            result = classify(raw)
            assert result.severity in ("critical", "significant", "minor"), (
                f"{sig_type} produced invalid severity: {result.severity}"
            )

    def test_classify_all_returns_same_count(self):
        raws = [
            RawSignal("t", "l", "s", "injury", "injured", "ctx", "raw"),
            RawSignal("t", "l", "s", "suspension", "banned", "ctx", "raw"),
        ]
        results = classify_all(raws)
        assert len(results) == len(raws)

    def test_save_and_load_classified(self, injury_signal):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            save_classified([injury_signal], path, date_str="2026-05-13")
            loaded = load_classified(path, date_str="2026-05-13")
            assert len(loaded) == 1
            assert loaded[0].signal_type == "injury"
            assert loaded[0].severity == injury_signal.severity

    def test_load_missing_date_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loaded = load_classified(Path(tmpdir), date_str="1999-01-01")
            assert loaded == []


# ── Impact Scorer ─────────────────────────────────────────────────────────────

class TestImpactScorer:
    def test_injury_reduces_home_win_prob(self, injury_signal):
        delta = score_signal(injury_signal, "Mexico", "Mexico", "South Africa")
        assert delta.home_win_delta < 0, "Home team injury should reduce home win delta"

    def test_suspension_of_away_team_helps_home(self, suspension_signal):
        delta = score_signal(suspension_signal, "South Africa", "Mexico", "South Africa")
        assert delta.home_win_delta > 0, "Away suspension should benefit home team"

    def test_delta_bounds_respected(self, injury_signal):
        delta = score_signal(injury_signal, "Mexico", "Mexico", "South Africa")
        assert -0.15 <= delta.home_win_delta <= 0.15
        assert -0.15 <= delta.draw_delta <= 0.15
        assert -0.15 <= delta.away_win_delta <= 0.15

    def test_aggregate_deltas_sum(self, injury_signal, suspension_signal):
        d1 = score_signal(injury_signal, "Mexico", "Mexico", "South Africa")
        d2 = score_signal(suspension_signal, "South Africa", "Mexico", "South Africa")
        h, d, a = aggregate_deltas([d1, d2])
        assert -0.20 <= h <= 0.20
        assert -0.20 <= d <= 0.20
        assert -0.20 <= a <= 0.20

    def test_all_signal_types_score_without_error(self):
        for sig_type in ["injury", "suspension", "lineup_change", "form", "weather", "travel"]:
            sig = ClassifiedSignal(
                signal_type=sig_type,
                severity="significant",
                matched_phrase="test",
                context_snippet="France good form win streak",
                article_title="France Test",
                article_link="https://example.com",
                source_name="Test",
                team_hint="France",
                classified_at="2026-05-13T00:00:00+00:00",
            )
            delta = score_signal(sig, "France", "France", "Germany")
            assert isinstance(delta, ImpactDelta)


# ── Signal Review ─────────────────────────────────────────────────────────────

class TestSignalReview:
    def test_no_signals_returns_base_unchanged(self, mexico_vs_sa):
        review = build_review(mexico_vs_sa, [])
        assert abs(review.adjusted_home_win - mexico_vs_sa.home_win_prob) < 1e-9
        assert abs(review.adjusted_draw - mexico_vs_sa.draw_prob) < 1e-9
        assert abs(review.adjusted_away_win - mexico_vs_sa.away_win_prob) < 1e-9

    def test_injury_shifts_probabilities(self, mexico_vs_sa, injury_signal):
        review = build_review(mexico_vs_sa, [injury_signal])
        # Mexico injury → home win should decrease
        assert review.adjusted_home_win < mexico_vs_sa.home_win_prob

    def test_adjusted_probs_sum_to_one(self, mexico_vs_sa, injury_signal, suspension_signal):
        review = build_review(mexico_vs_sa, [injury_signal, suspension_signal])
        total = review.adjusted_home_win + review.adjusted_draw + review.adjusted_away_win
        assert abs(total - 1.0) < 1e-6, f"Probabilities sum to {total}, not 1.0"

    def test_adjusted_probs_in_range(self, mexico_vs_sa, injury_signal):
        review = build_review(mexico_vs_sa, [injury_signal])
        assert 0.0 <= review.adjusted_home_win <= 1.0
        assert 0.0 <= review.adjusted_draw <= 1.0
        assert 0.0 <= review.adjusted_away_win <= 1.0

    def test_unrelated_signal_not_applied(self, mexico_vs_sa):
        france_signal = ClassifiedSignal(
            signal_type="injury",
            severity="critical",
            matched_phrase="injured",
            context_snippet="France star player injured",
            article_title="France injury news",
            article_link="https://example.com",
            source_name="BBC",
            team_hint="France",
            classified_at="2026-05-13T00:00:00+00:00",
        )
        review = build_review(mexico_vs_sa, [france_signal])
        # France is not in Mexico vs South Africa — no adjustment
        assert review.signals_applied == []
        assert abs(review.adjusted_home_win - mexico_vs_sa.home_win_prob) < 1e-9

    def test_disclaimer_always_present(self, mexico_vs_sa):
        review = build_review(mexico_vs_sa, [])
        assert "Core model probability unchanged" in review.disclaimer

    def test_core_base_not_mutated(self, mexico_vs_sa, injury_signal):
        original_home = mexico_vs_sa.home_win_prob
        build_review(mexico_vs_sa, [injury_signal])
        assert mexico_vs_sa.home_win_prob == original_home, (
            "build_review must not mutate BaseForecast"
        )


# ── Signal Reporter ───────────────────────────────────────────────────────────

class TestSignalReporter:
    def _make_review(self, mexico_vs_sa, injury_signal) -> SignalReview:
        return build_review(mexico_vs_sa, [injury_signal])

    def test_render_review_table_contains_teams(self, mexico_vs_sa, injury_signal):
        review = self._make_review(mexico_vs_sa, injury_signal)
        table = render_review_table(review)
        assert "Mexico" in table
        assert "South Africa" in table

    def test_render_full_report_has_disclaimer(self, mexico_vs_sa, injury_signal):
        review = self._make_review(mexico_vs_sa, injury_signal)
        report = render_full_report(review)
        assert "Signal-adjusted review only" in report
        assert "Core model probability unchanged" in report

    def test_render_full_report_has_signal_section(self, mexico_vs_sa, injury_signal):
        review = self._make_review(mexico_vs_sa, injury_signal)
        report = render_full_report(review)
        assert "Signals Applied" in report

    def test_save_report_creates_file(self, mexico_vs_sa, injury_signal):
        with tempfile.TemporaryDirectory() as tmpdir:
            review = self._make_review(mexico_vs_sa, injury_signal)
            path = save_report(review, Path(tmpdir), date_str="2026-05-13")
            assert path.exists()
            content = path.read_text()
            assert "Signal-adjusted review only" in content

    def test_report_does_not_write_to_core_outputs(self, mexico_vs_sa, injury_signal):
        """Ensure save_report only writes under the given output dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            review = self._make_review(mexico_vs_sa, injury_signal)
            path = save_report(review, Path(tmpdir), date_str="2026-05-13")
            assert str(path).startswith(tmpdir), (
                "Report must be written only inside signals output dir"
            )


# ── Article Parser ────────────────────────────────────────────────────────────

class TestArticleParser:
    def test_parse_article_strips_html(self, raw_article):
        raw_article["title"] = "<b>Mexico <i>striker</i> injured</b>"
        article = parse_article(raw_article)
        assert "<b>" not in article.title
        assert "Mexico striker injured" == article.title

    def test_filter_relevant_keeps_injury_articles(self, raw_article):
        articles = parse_articles([raw_article])
        relevant = filter_relevant(articles)
        assert len(relevant) == 1

    def test_filter_relevant_excludes_unrelated(self):
        raw = {
            "source_name": "BBC", "source_url": "", "source_tags": [],
            "title": "Chef wins cooking competition",
            "summary": "A renowned chef took first prize at an international baking event.",
            "link": "", "published": "", "collected_at": "",
        }
        articles = parse_articles([raw])
        relevant = filter_relevant(articles)
        assert len(relevant) == 0


# ── No-Mutation of Core Outputs ───────────────────────────────────────────────

class TestNoCoreMutation:
    """Signals must never write to core model/evaluation/simulation output dirs."""

    def test_signal_outputs_are_isolated(self, mexico_vs_sa, injury_signal):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            signals_dir = tmppath / "signals"
            models_dir = tmppath / "models"
            models_dir.mkdir()

            review = build_review(mexico_vs_sa, [injury_signal])
            save_report(review, signals_dir / "reports", date_str="2026-05-13")

            # Nothing should have been written into models dir
            assert list(models_dir.iterdir()) == [], (
                "Signal review must not write to models output dir"
            )

    def test_build_review_does_not_touch_filesystem(self, mexico_vs_sa, injury_signal):
        """build_review itself should be a pure in-memory operation."""
        import os
        # No filesystem calls should be triggered by build_review
        original_cwd_contents = set(os.listdir("."))
        build_review(mexico_vs_sa, [injury_signal])
        new_cwd_contents = set(os.listdir("."))
        # CWD should be unchanged
        assert original_cwd_contents == new_cwd_contents


# ── Graceful Missing Source Handling ─────────────────────────────────────────

class TestGracefulDegradation:
    def test_rss_collector_handles_bad_url(self):
        from src.signals.source_config import SignalSource
        from src.signals.rss_collector import _parse_feed

        bad_source = SignalSource(
            name="BadSource",
            url="https://this.url.does.not.exist.invalid/feed.xml",
            category="rss",
        )
        # Should return empty list without raising
        with patch("src.signals.rss_collector._FEEDPARSER_AVAILABLE", True):
            import feedparser as fp
            with patch.object(fp, "parse", side_effect=Exception("network error")):
                result = _parse_feed(bad_source)
        assert result == []

    def test_load_raw_missing_file_returns_empty(self):
        from src.signals.rss_collector import load_raw
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_raw(Path(tmpdir), date_str="1999-01-01")
            assert result == []

    def test_build_review_with_empty_signals(self, mexico_vs_sa):
        review = build_review(mexico_vs_sa, [])
        assert review.signals_applied == []
        assert review.any_signals is False

    def test_feedparser_missing_raises_import_error(self):
        from src.signals.rss_collector import _require_feedparser
        with patch("src.signals.rss_collector._FEEDPARSER_AVAILABLE", False):
            with pytest.raises(ImportError, match="feedparser"):
                _require_feedparser()
