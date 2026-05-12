from __future__ import annotations


def validate_weights(elo: float, poisson: float, xgboost: float, tol: float = 1e-6) -> None:
    """Raise ValueError if weights are invalid (negative or don't sum to 1)."""
    for name, w in [("elo", elo), ("poisson", poisson), ("xgboost", xgboost)]:
        if w < 0:
            raise ValueError(f"Weight '{name}' must be non-negative, got {w}")
    total = elo + poisson + xgboost
    if abs(total - 1.0) > tol:
        raise ValueError(f"Weights must sum to 1.0, got {total:.8f}")


def simplex_grid(resolution: int = 5) -> list[tuple[float, float, float]]:
    """Generate a uniform grid of (elo, poisson, xgboost) weight triples on the 3-simplex."""
    step = 1.0 / resolution
    points: list[tuple[float, float, float]] = []
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            k = resolution - i - j
            points.append((round(i * step, 6), round(j * step, 6), round(k * step, 6)))
    return points
