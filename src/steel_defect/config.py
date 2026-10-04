"""Central defaults. Commands expose the useful experiment settings."""
from dataclasses import asdict, dataclass
from pathlib import Path

CLASSES = ("crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches")
DEFAULT_DATA_DIR = Path("data/raw/NEU-CLS")
DEFAULT_MANIFEST = Path("data/processed/manifest.csv")
DEFAULT_RUN_DIR = Path("outputs")
DEFAULT_SPLIT_SEED = 42
DEFAULT_VAL_FRACTION = 0.10
DEFAULT_TEST_FRACTION = 0.10
SPLIT_METHOD = "pooled_stratified_v2"


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    image_size: int = 128
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    dropout: float = 0.3
    patience: int = 8
    filters: tuple[int, ...] = (32, 64, 128)
    dense_units: int = 64
    rotation: float = 0.025  # Fraction of a full turn: at most +/-9 degrees.
    translation: float = 0.05
    zoom: float = 0.05

    def __post_init__(self):
        if self.image_size < 32 or self.batch_size < 1 or self.epochs < 1 or self.patience < 1:
            raise ValueError("Use image_size >= 32 and positive batch_size, epochs and patience.")
        if self.learning_rate <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("Learning rate must be positive and dropout must be in [0, 1).")

    def to_dict(self):
        return asdict(self)
