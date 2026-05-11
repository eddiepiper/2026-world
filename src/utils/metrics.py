import numpy as np


def brier_score(predicted_probs: list[float], outcomes: list[int]) -> float:
    """Lower is better. outcome=1 if event occurred, 0 otherwise."""
    return float(np.mean([(p - o) ** 2 for p, o in zip(predicted_probs, outcomes)]))


def accuracy(predictions: list[str], actuals: list[str]) -> float:
    if not predictions:
        return 0.0
    correct = sum(p == a for p, a in zip(predictions, actuals))
    return correct / len(predictions)
