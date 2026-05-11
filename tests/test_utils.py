import pytest

from src.utils.helpers import safe_div, normalize_probabilities, clamp
from src.utils.metrics import brier_score, accuracy


# --- helpers ---

def test_safe_div_normal():
    assert safe_div(10.0, 2.0) == 5.0


def test_safe_div_zero_denominator_returns_default():
    assert safe_div(5.0, 0.0, default=99.0) == 99.0


def test_safe_div_zero_denominator_default_is_one():
    assert safe_div(5.0, 0.0) == 1.0


def test_normalize_probabilities_sums_to_one():
    result = normalize_probabilities([0.3, 0.4, 0.3])
    assert abs(sum(result) - 1.0) < 1e-9


def test_normalize_probabilities_proportional():
    result = normalize_probabilities([1.0, 2.0, 1.0])
    assert abs(result[1] - 0.5) < 1e-9


def test_normalize_probabilities_zero_total_equal_split():
    result = normalize_probabilities([0.0, 0.0, 0.0])
    assert all(abs(p - 1 / 3) < 1e-9 for p in result)


def test_clamp_within_range():
    assert clamp(0.5) == 0.5


def test_clamp_below_min():
    assert clamp(-0.1) == 0.0


def test_clamp_above_max():
    assert clamp(1.5) == 1.0


def test_clamp_custom_range():
    assert clamp(5.0, min_val=0.0, max_val=3.0) == 3.0


# --- metrics ---

def test_brier_score_perfect():
    assert brier_score([1.0, 1.0], [1, 1]) == 0.0


def test_brier_score_worst():
    score = brier_score([0.0, 0.0], [1, 1])
    assert abs(score - 1.0) < 1e-9


def test_brier_score_partial():
    score = brier_score([0.5], [1])
    assert abs(score - 0.25) < 1e-9


def test_accuracy_all_correct():
    assert accuracy(["A", "B", "C"], ["A", "B", "C"]) == 1.0


def test_accuracy_none_correct():
    assert accuracy(["A", "B"], ["C", "D"]) == 0.0


def test_accuracy_partial():
    assert accuracy(["A", "B", "C", "D"], ["A", "X", "C", "X"]) == 0.5


def test_accuracy_empty():
    assert accuracy([], []) == 0.0
