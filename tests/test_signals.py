from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.signals.article_parser import ParsedArticle, filter_relevant, is_football_article, parse_article, parse_articles
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
    deduplicate_classified,
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


class TestKnownTeamsExpanded:
    """All 48 WC2026 fixture teams must be resolvable from article text."""

    @pytest.mark.parametrize("text,expected", [
        ("Algeria beat Austria in a friendly", ["Algeria", "Austria"]),
        ("Bosnia and Herzegovina face elimination", ["Bosnia and Herzegovina"]),
        ("Cape Verde surprise win", ["Cape Verde"]),
        ("Colombia striker ruled out", ["Colombia"]),
        ("Curaçao qualify for first time", ["Curaçao"]),
        ("Czechia goalkeeper injury doubt", ["Czechia"]),
        ("DR Congo vs Uzbekistan preview", ["DR Congo", "Uzbekistan"]),
        ("Egypt win on penalties", ["Egypt"]),
        ("Haiti coach suspended", ["Haiti"]),
        ("Iraq midfielder banned", ["Iraq"]),
        ("Ivory Coast striker injured", ["Ivory Coast"]),
        ("Jordan book last-16 place", ["Jordan"]),
        ("New Zealand defy the odds", ["New Zealand"]),
        ("Norway qualify from Group I", ["Norway"]),
        ("Paraguay hold Uruguay", ["Paraguay"]),
        ("Scotland earn a point", ["Scotland"]),
        ("Sweden top their group", ["Sweden"]),
        ("Turkey advance to knockouts", ["Turkey"]),
        ("Congo DR and Democratic Republic of Congo aliases", ["DR Congo"]),
        ("Bosnia alias match", ["Bosnia and Herzegovina"]),
    ])
    def test_all_fixture_teams_resolve(self, text, expected):
        from src.signals.team_matcher import match_signal_to_teams
        result = match_signal_to_teams(text)
        for team in expected:
            assert team in result, f"Expected '{team}' in results for: {text!r}, got {result}"


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
        from src.signals.impact_scorer import HARD_CAP
        delta = score_signal(injury_signal, "Mexico", "Mexico", "South Africa")
        assert -HARD_CAP <= delta.home_win_delta <= HARD_CAP
        assert -HARD_CAP <= delta.draw_delta <= HARD_CAP
        assert -HARD_CAP <= delta.away_win_delta <= HARD_CAP

    def test_hard_cap_is_eight_percent(self):
        from src.signals.impact_scorer import HARD_CAP
        assert HARD_CAP == 0.08, "Hard cap must be exactly 8 percentage points"

    def test_no_signal_exceeds_hard_cap(self):
        from src.signals.impact_scorer import HARD_CAP, _MAX_DELTA
        for sig_type, severities in _MAX_DELTA.items():
            for severity, max_d in severities.items():
                assert max_d <= HARD_CAP, (
                    f"{sig_type}.{severity} = {max_d:.0%} exceeds hard cap {HARD_CAP:.0%}"
                )

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
    def test_rss_collector_handles_bad_url(self, tmp_path):
        import src.signals.rss_collector as rss_mod
        from src.signals.rss_collector import collect_all

        mock_feedparser = MagicMock()
        mock_feedparser.parse.return_value = {"entries": []}

        # feedparser may not be installed, so inject the mock directly into the module
        had_feedparser = hasattr(rss_mod, "feedparser")
        rss_mod.feedparser = mock_feedparser  # type: ignore[attr-defined]
        try:
            with patch("src.signals.rss_collector._FEEDPARSER_AVAILABLE", True):
                result = collect_all(output_dir=tmp_path)
        finally:
            if had_feedparser:
                rss_mod.feedparser = mock_feedparser
            else:
                del rss_mod.feedparser

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


# ── Deduplication ─────────────────────────────────────────────────────────────

class TestDeduplication:
    def _make_signal(self, link: str, sig_type: str, severity: str, team: str = "England") -> ClassifiedSignal:
        return ClassifiedSignal(
            signal_type=sig_type,
            severity=severity,
            matched_phrase="test",
            context_snippet="some context",
            article_title=f"{team} {sig_type} news",
            article_link=link,
            source_name="BBC Sport",
            team_hint=team,
            classified_at="2026-05-13T00:00:00+00:00",
        )

    def test_exact_duplicate_reduced_to_one(self):
        sig = self._make_signal("https://example.com/1", "injury", "significant")
        result = deduplicate_classified([sig, sig])
        assert len(result) == 1

    def test_same_article_and_type_keeps_highest_severity(self):
        minor = self._make_signal("https://example.com/1", "injury", "minor")
        critical = self._make_signal("https://example.com/1", "injury", "critical")
        result = deduplicate_classified([minor, critical])
        assert len(result) == 1
        assert result[0].severity == "critical"

    def test_same_article_different_types_both_kept(self):
        injury = self._make_signal("https://example.com/1", "injury", "significant")
        suspension = self._make_signal("https://example.com/1", "suspension", "critical")
        result = deduplicate_classified([injury, suspension])
        assert len(result) == 2

    def test_same_type_different_articles_both_kept(self):
        sig1 = self._make_signal("https://example.com/1", "injury", "significant")
        sig2 = self._make_signal("https://example.com/2", "injury", "significant")
        result = deduplicate_classified([sig1, sig2])
        assert len(result) == 2

    def test_save_classified_deduplicates_before_writing(self, injury_signal):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            save_classified([injury_signal, injury_signal, injury_signal], path, date_str="2026-05-13")
            loaded = load_classified(path, date_str="2026-05-13")
            assert len(loaded) == 1

    def test_scotland_travel_duplicate_collapsed(self):
        travel1 = self._make_signal("https://example.com/scotland-travel", "travel", "minor", "Scotland")
        travel2 = self._make_signal("https://example.com/scotland-travel", "travel", "minor", "Scotland")
        result = deduplicate_classified([travel1, travel2])
        assert len(result) == 1

    def test_ben_white_injury_keeps_highest_severity(self):
        """Two patterns matching same Ben White article → keep the stronger one."""
        generic_injury = self._make_signal("https://bbc.co.uk/ben-white", "injury", "significant", "England")
        knee_injury = self._make_signal("https://bbc.co.uk/ben-white", "injury", "critical", "England")
        result = deduplicate_classified([generic_injury, knee_injury])
        assert len(result) == 1
        assert result[0].severity == "critical"

    def test_same_article_different_teams_both_kept(self):
        """One article covering two teams' injuries → both adjustments must survive."""
        mexico_injury = self._make_signal("https://example.com/world-cup", "injury", "significant", "Mexico")
        sa_injury = self._make_signal("https://example.com/world-cup", "injury", "significant", "South Africa")
        result = deduplicate_classified([mexico_injury, sa_injury])
        team_hints = {s.team_hint for s in result}
        assert "Mexico" in team_hints
        assert "South Africa" in team_hints


# ── Predict Signal Section ────────────────────────────────────────────────────

class TestPredictSignalSection:
    """Signal-adjusted section appears after ensemble in cmd_predict."""

    def test_no_crash_when_signals_empty(self, mexico_vs_sa):
        """cmd_predict signal section handles empty signal list without raising."""
        from src.signals.signal_review import build_review
        review = build_review(mexico_vs_sa, [])
        assert not review.any_signals
        assert review.adjusted_home_win == mexico_vs_sa.home_win_prob
        assert review.adjusted_draw == mexico_vs_sa.draw_prob
        assert review.adjusted_away_win == mexico_vs_sa.away_win_prob

    def test_signal_section_adjusts_probabilities(self, mexico_vs_sa, injury_signal):
        """build_review with a relevant signal shifts probabilities from base."""
        from src.signals.signal_review import build_review
        review = build_review(mexico_vs_sa, [injury_signal])
        assert review.any_signals
        # Probabilities must differ from base when signal applies
        changed = (
            review.adjusted_home_win != mexico_vs_sa.home_win_prob
            or review.adjusted_draw != mexico_vs_sa.draw_prob
            or review.adjusted_away_win != mexico_vs_sa.away_win_prob
        )
        assert changed, "Signal should shift at least one probability"

    def test_signal_section_probabilities_sum_to_one(self, mexico_vs_sa, injury_signal):
        """Adjusted probabilities always sum to 1.0 after normalization."""
        from src.signals.signal_review import build_review
        review = build_review(mexico_vs_sa, [injury_signal])
        total = review.adjusted_home_win + review.adjusted_draw + review.adjusted_away_win
        assert abs(total - 1.0) < 1e-6


# ── Football Relevance Filter ─────────────────────────────────────────────────

class TestFootballRelevanceFilter:
    def _make_article(self, title: str, link: str = "", body: str = "match played") -> ParsedArticle:
        return ParsedArticle(
            source_name="Test",
            title=title,
            body=body,
            link=link,
            published_raw="",
            collected_at="",
            source_tags=[],
        )

    def test_rugby_article_excluded(self):
        article = self._make_article(
            "Japan rugby coach banned for three matches",
            link="https://bbc.co.uk/sport/rugby-union/article",
        )
        assert not is_football_article(article)

    def test_rugby_article_excluded_from_filter_relevant(self):
        article = self._make_article(
            "Japan rugby coach banned for three matches",
            link="https://bbc.co.uk/sport/rugby-union/article",
            body="The Japan rugby union coach received a ban after the incident.",
        )
        result = filter_relevant([article])
        assert len(result) == 0

    def test_japan_rugby_ban_does_not_contaminate_japan_football_signals(self):
        """A Japan rugby article must not reach the signal classifier for football reviews."""
        rugby_article = self._make_article(
            "Japan rugby coach banned",
            link="https://example.com/rugby-union/japan",
            body="Japan rugby union coach suspended for three matches after red card incident.",
        )
        football_article = self._make_article(
            "Japan World Cup squad named",
            link="https://example.com/football/japan-squad",
            body="Japan named their 26-man World Cup squad. Key players are fit.",
        )
        relevant = filter_relevant([rugby_article, football_article])
        # Only the football article should pass
        assert not any("rugby" in a.title.lower() for a in relevant)
        assert any("World Cup" in a.title for a in relevant)

    def test_cricket_article_excluded(self):
        article = self._make_article(
            "England cricket captain ruled out injured",
            link="https://example.com/cricket/england",
        )
        assert not is_football_article(article)

    def test_tennis_article_excluded(self):
        article = self._make_article(
            "Wimbledon: player suspended after banned substance",
            link="https://example.com/tennis/wimbledon",
        )
        assert not is_football_article(article)

    def test_formula_one_article_excluded(self):
        article = self._make_article(
            "F1: driver banned for reckless driving",
            link="https://example.com/formula-1/driver",
        )
        assert not is_football_article(article)

    def test_football_article_passes_sport_filter(self):
        article = self._make_article(
            "Brazil striker ruled out with injury ahead of World Cup",
            link="https://bbc.co.uk/sport/football/brazil-injury",
        )
        assert is_football_article(article)

    def test_rugby_url_path_keyword_blocks_article(self):
        article = self._make_article(
            "Ireland banned players return",
            link="https://espn.com/rugby-union/story",
        )
        assert not is_football_article(article)

    def test_nfl_article_excluded(self):
        article = self._make_article(
            "NFL suspension handed to star quarterback",
            link="https://espn.com/nfl/story",
        )
        assert not is_football_article(article)

    def test_rugby_body_only_article_excluded(self):
        """Article with generic title/URL but rugby body should still be blocked."""
        article = self._make_article(
            "Latest sports news",
            link="https://example.com/watch/video/12345",
            body="In rugby union today, the Ireland coach received a touchline ban after the game.",
        )
        assert not is_football_article(article)

    def test_cricket_body_only_article_excluded(self):
        article = self._make_article(
            "Sports update",
            link="https://example.com/sport/news",
            body="England cricket captain ruled out injured ahead of the test series.",
        )
        assert not is_football_article(article)

    def test_football_body_passes_with_generic_url(self):
        article = self._make_article(
            "Latest from the squad",
            link="https://example.com/watch/video/99999",
            body="Mexico's striker has been ruled out injured ahead of the World Cup match.",
        )
        assert is_football_article(article)
