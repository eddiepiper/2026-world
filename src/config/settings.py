from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent


@dataclass
class EloConfig:
    k_factor: float = 32.0
    home_advantage: float = 100.0
    initial_rating: float = 1500.0


@dataclass
class PoissonConfig:
    home_advantage: float = 1.2
    default_attack: float = 1.0
    default_defense: float = 1.0
    avg_goals_fallback: float = 1.5


@dataclass
class SimulationConfig:
    n_simulations: int = 10_000
    random_seed: int | None = None


@dataclass
class PredictionWeights:
    elo: float = 0.5
    poisson: float = 0.5


@dataclass
class Settings:
    elo: EloConfig = field(default_factory=EloConfig)
    poisson: PoissonConfig = field(default_factory=PoissonConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    weights: PredictionWeights = field(default_factory=PredictionWeights)
    data_dir: Path = field(default_factory=lambda: BASE_DIR / "data")
    outputs_dir: Path = field(default_factory=lambda: BASE_DIR / "outputs")


settings = Settings()
