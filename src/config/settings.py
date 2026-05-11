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
class DataConfig:
    start_year: int = 1980
    results_url: str = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    goalscorers_url: str = "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv"
    shootouts_url: str = "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"
    team_name_map: dict[str, str] = field(default_factory=lambda: {
        "Korea Republic": "South Korea",
        "IR Iran": "Iran",
        "USA": "United States",
        "Türkiye": "Turkey",
        "Côte d'Ivoire": "Ivory Coast",
        "China PR": "China",
        "DR Congo": "Congo DR",
    })


@dataclass
class FeatureConfig:
    form_window: int = 5


@dataclass
class MLConfig:
    test_split_date: str = "2020-01-01"
    random_seed: int = 42
    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    model_output_dir: str = "outputs/models"


@dataclass
class Settings:
    elo: EloConfig = field(default_factory=EloConfig)
    poisson: PoissonConfig = field(default_factory=PoissonConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    weights: PredictionWeights = field(default_factory=PredictionWeights)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    data_dir: Path = field(default_factory=lambda: BASE_DIR / "data")
    outputs_dir: Path = field(default_factory=lambda: BASE_DIR / "outputs")


settings = Settings()
