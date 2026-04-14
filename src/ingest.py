"""Phase 1 — Data ingestion and preprocessing."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    clean_eval_loader: DataLoader
    noisy_eval_loader: DataLoader
    d: int
    k: int
    metadata: dict
    flip_mask: np.ndarray | None
    scaler: StandardScaler | MinMaxScaler
    X_noisy_scaled: np.ndarray
    X_clean_scaled: np.ndarray
    y_clean: np.ndarray


def ingest(npz_path: str, config: PipelineConfig) -> DatasetBundle:
    """Load .npz, scale features, return DataLoaders + raw arrays."""
    data = np.load(npz_path, allow_pickle=True)

    X_noisy = data["X_noisy"]
    y_noisy = data["y_noisy"]
    X_clean = data["X_clean"]
    y_clean = data["y_clean"]
    # Metadata: embedded in .npz or sidecar .json
    if "metadata" in data:
        metadata = data["metadata"].item() if data["metadata"].ndim == 0 else dict(data["metadata"])
    else:
        json_path = Path(npz_path).with_suffix(".json")
        if json_path.exists():
            with open(json_path) as f:
                raw = json.load(f)
            metadata = {
                "topology": raw.get("topology", "unknown"),
                "noise_type": raw.get("feature_noise", {}).get("distribution", "unknown"),
                "noise_scale": raw.get("feature_noise", {}).get("scale", 0.0),
                "n_samples": raw.get("params", {}).get("n", len(X_noisy)),
                "n_classes": len(np.unique(y_noisy)),
                "filename": Path(npz_path).name,
            }
        else:
            metadata = {
                "topology": "unknown", "noise_type": "unknown", "noise_scale": 0.0,
                "n_samples": len(X_noisy), "n_classes": len(np.unique(y_noisy)),
                "filename": Path(npz_path).name,
            }
    flip_mask = data["flip_mask"] if "flip_mask" in data else None

    d = X_noisy.shape[1]
    k = len(np.unique(y_noisy))

    if d > 30:
        logger.warning("d=%d exceeds 30 — VRAM usage may be high", d)

    # Fit scaler on noisy only, transform both
    if config.scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    X_noisy_scaled = scaler.fit_transform(X_noisy)
    X_clean_scaled = scaler.transform(X_clean)

    # Tensors — CPU only, device transfer happens in engine
    X_noisy_t = torch.tensor(X_noisy_scaled, dtype=torch.float32)
    y_noisy_t = torch.tensor(y_noisy, dtype=torch.long)
    X_clean_t = torch.tensor(X_clean_scaled, dtype=torch.float32)
    y_clean_t = torch.tensor(y_clean, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_noisy_t, y_noisy_t),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    clean_eval_loader = DataLoader(
        TensorDataset(X_clean_t, y_clean_t),
        batch_size=config.batch_size,
        shuffle=False,
    )
    noisy_eval_loader = DataLoader(
        TensorDataset(X_noisy_t, y_noisy_t),
        batch_size=config.batch_size,
        shuffle=False,
    )

    return DatasetBundle(
        train_loader=train_loader,
        clean_eval_loader=clean_eval_loader,
        noisy_eval_loader=noisy_eval_loader,
        d=d,
        k=k,
        metadata=metadata,
        flip_mask=flip_mask,
        scaler=scaler,
        X_noisy_scaled=X_noisy_scaled,
        X_clean_scaled=X_clean_scaled,
        y_clean=y_clean,
    )
