"""Phase 3 — Training loop, evaluation, RBF center init, and teardown."""

import gc
import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist as scipy_cdist

from config import PipelineConfig
from src.ingest import DatasetBundle
from src.models import RBFClassifier

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    accuracy: float
    loss: float
    logits: torch.Tensor  # (N, k) on CPU


@dataclass
class TrainResult:
    mlp_loss_curve: list[float] = field(default_factory=list)
    rbf_loss_curve: list[float] = field(default_factory=list)
    mlp_evals: dict[str, EvalResult] = field(default_factory=dict)
    rbf_evals: dict[str, EvalResult] = field(default_factory=dict)


def init_rbf_centers(rbf: RBFClassifier, X_np: np.ndarray, config: PipelineConfig):
    """Phase 3A — Initialize RBF centers via MiniBatchKMeans."""
    n_centers = min(config.rbf_n_centers, len(X_np))
    kmeans = MiniBatchKMeans(n_clusters=n_centers, batch_size=1024, n_init=3)
    kmeans.fit(X_np)
    centers = kmeans.cluster_centers_

    # Sigma = average nearest-neighbor distance between centers
    dists = scipy_cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    sigma = dists.min(axis=1).mean()

    rbf.load_centers(centers, sigma)
    logger.info("RBF centers initialized: sigma=%.4f", sigma)


def evaluate(model: nn.Module, loader, device: str) -> EvalResult:
    """Phase 3C — Evaluate model, return accuracy + loss + full logits."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            n_samples += len(y_batch)
            all_logits.append(logits.cpu())
            all_labels.append(y_batch.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    preds = all_logits.argmax(dim=1)
    accuracy = (preds == all_labels).float().mean().item()
    avg_loss = total_loss / n_samples

    return EvalResult(accuracy=accuracy, loss=avg_loss, logits=all_logits)


def train(mlp: nn.Module, rbf: nn.Module, bundle: DatasetBundle, config: PipelineConfig) -> TrainResult:
    """Phase 3B — Train both models side-by-side, then evaluate."""
    device = config.device
    mlp.to(device)
    rbf.to(device)

    criterion = nn.CrossEntropyLoss()
    mlp_opt = torch.optim.AdamW(mlp.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    rbf_opt = torch.optim.AdamW(rbf.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    result = TrainResult()

    for epoch in range(config.epochs):
        mlp.train()
        rbf.train()
        mlp_epoch_loss = 0.0
        rbf_epoch_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in bundle.train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # MLP
            mlp_opt.zero_grad()
            mlp_loss = criterion(mlp(X_batch), y_batch)
            mlp_loss.backward()
            mlp_opt.step()

            # RBF
            rbf_opt.zero_grad()
            rbf_loss = criterion(rbf(X_batch), y_batch)
            rbf_loss.backward()
            rbf_opt.step()

            mlp_epoch_loss += mlp_loss.item()
            rbf_epoch_loss += rbf_loss.item()
            n_batches += 1

        result.mlp_loss_curve.append(mlp_epoch_loss / n_batches)
        result.rbf_loss_curve.append(rbf_epoch_loss / n_batches)

    # Evaluate on both splits
    result.mlp_evals["noisy"] = evaluate(mlp, bundle.noisy_eval_loader, device)
    result.mlp_evals["clean"] = evaluate(mlp, bundle.clean_eval_loader, device)
    result.rbf_evals["noisy"] = evaluate(rbf, bundle.noisy_eval_loader, device)
    result.rbf_evals["clean"] = evaluate(rbf, bundle.clean_eval_loader, device)

    return result


def cleanup(*objects):
    """Phase 3D — Delete objects and free GPU memory."""
    for obj in objects:
        del obj
    torch.cuda.empty_cache()
    gc.collect()
