def safe_div(numerator: float, denominator: float, default: float = 1.0) -> float:
    return numerator / denominator if denominator != 0 else default


def normalize_probabilities(probs: list[float]) -> list[float]:
    total = sum(probs)
    if total == 0:
        n = len(probs)
        return [1.0 / n] * n
    return [p / total for p in probs]


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    return max(min_val, min(max_val, value))
