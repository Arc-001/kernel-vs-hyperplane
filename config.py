from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class PipelineConfig:
    # Hardware
    device: str = ""  # auto-detected if empty

    # Data
    batch_size: int = 256
    scaler_type: str = "standard"  # "standard" | "minmax"

    # MLP
    mlp_hidden_layers: tuple[int, ...] = (128, 128)
    mlp_activation: str = "relu"  # "relu" | "gelu"

    # RBF
    rbf_n_centers: int = 256
    rbf_sigma_init: str = "auto"  # "auto" = avg inter-center distance
    rbf_sigma_learnable: bool = True

    # Training
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100

    # Visualization
    mesh_resolution: int = 200  # 200x200 grid for decision boundaries
    inference_batch_size: int = 1024  # mesh grid / eval batching

    # Paths
    data_dir: str = "data/generated"
    output_dir: str = "outputs"

    def __post_init__(self):
        if not self.device:
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
            object.__setattr__(self, "device", resolved)

    @property
    def metrics_dir(self) -> Path:
        return Path(self.output_dir) / "metrics"

    @property
    def figures_dir(self) -> Path:
        return Path(self.output_dir) / "figures"


def get_config(**overrides) -> PipelineConfig:
    """Factory with overrides for easy testing and runner customization."""
    return PipelineConfig(**overrides)
