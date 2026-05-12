from __future__ import annotations

import pytest

from src.optimization.weight_search import validate_weights, simplex_grid


class TestValidateWeights:
    def test_valid_weights_pass(self):
        validate_weights(0.3, 0.3, 0.4)  # should not raise

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            validate_weights(-0.1, 0.5, 0.6)

    def test_sum_not_one_raises(self):
        with pytest.raises(ValueError, match="sum to 1"):
            validate_weights(0.2, 0.2, 0.2)

    def test_tolerance_boundary(self):
        validate_weights(1/3, 1/3, 1/3)  # floating point near 1.0, should pass

    def test_all_zero_except_one(self):
        validate_weights(0.0, 0.0, 1.0)  # edge: full weight on xgboost


class TestSimplexGrid:
    def test_returns_list_of_triples(self):
        grid = simplex_grid(resolution=2)
        assert all(len(p) == 3 for p in grid)

    def test_all_weights_non_negative(self):
        for elo, poisson, xgboost in simplex_grid(resolution=5):
            assert elo >= 0 and poisson >= 0 and xgboost >= 0

    def test_all_weights_sum_to_one(self):
        for elo, poisson, xgboost in simplex_grid(resolution=5):
            assert abs(elo + poisson + xgboost - 1.0) < 1e-6

    def test_resolution_2_has_correct_count(self):
        # For resolution=2: (2+1)*(2+2)/2 = 6 points
        grid = simplex_grid(resolution=2)
        assert len(grid) == 6

    def test_includes_corners(self):
        grid = simplex_grid(resolution=5)
        assert (1.0, 0.0, 0.0) in grid
        assert (0.0, 1.0, 0.0) in grid
        assert (0.0, 0.0, 1.0) in grid
