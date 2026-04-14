"""Phase 4 — Metric computation and JSON serialization."""

import json
import math
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import PipelineConfig
from src.engine import TrainResult
from src.ingest import DatasetBundle
from src.models import param_count


def compute_entropy(logits: torch.Tensor, k: int) -> tuple[float, float]:
    """Return (mean_entropy, normalized_entropy) from raw logits."""
    probs = F.softmax(logits, dim=1)
    entropy = -(probs * probs.clamp(min=1e-10).log()).sum(dim=1)
    mean = entropy.mean().item()
    normalized = mean / math.log(k)
    return mean, normalized


def compute_flip_metrics(
    noisy_logits: torch.Tensor,
    noisy_labels: np.ndarray,
    clean_labels: np.ndarray,
    flip_mask: np.ndarray | None,
) -> dict | None:
    """Compute flip-mask ablation metrics. Returns None if no flip_mask."""
    if flip_mask is None:
        return None
    flipped_idx = np.where(flip_mask)[0]
    if len(flipped_idx) == 0:
        return None
    preds = noisy_logits.argmax(dim=1).numpy()
    acc_on_flipped = (preds[flipped_idx] == clean_labels[flipped_idx]).mean()
    memorization_rate = (preds[flipped_idx] == noisy_labels[flipped_idx]).mean()
    return {
        "acc_on_flipped": float(acc_on_flipped),
        "flip_memorization_rate": float(memorization_rate),
    }


def _model_metrics(
    evals: dict, loss_curve: list[float], model, k: int,
    noisy_labels: np.ndarray, clean_labels: np.ndarray, flip_mask: np.ndarray | None,
) -> dict:
    """Build metric dict for one model."""
    noisy_ev = evals["noisy"]
    clean_ev = evals["clean"]
    ent_noisy_mean, ent_noisy_norm = compute_entropy(noisy_ev.logits, k)
    ent_clean_mean, ent_clean_norm = compute_entropy(clean_ev.logits, k)
    flip = compute_flip_metrics(noisy_ev.logits, noisy_labels, clean_labels, flip_mask)
    return {
        "noisy_acc": noisy_ev.accuracy,
        "clean_acc": clean_ev.accuracy,
        "perf_delta": noisy_ev.accuracy - clean_ev.accuracy,
        "noisy_loss": noisy_ev.loss,
        "clean_loss": clean_ev.loss,
        "mean_entropy_noisy": ent_noisy_mean,
        "mean_entropy_clean": ent_clean_mean,
        "normalized_entropy_noisy": ent_noisy_norm,
        "normalized_entropy_clean": ent_clean_norm,
        "flip_memorization_rate": flip["flip_memorization_rate"] if flip else None,
        "acc_on_flipped": flip["acc_on_flipped"] if flip else None,
        "param_count": param_count(model),
        "train_loss_curve": loss_curve,
    }


def compute_metrics(
    train_result: TrainResult, bundle: DatasetBundle, mlp, rbf, config: PipelineConfig,
) -> dict:
    """Build full JSON-serializable metrics dict."""
    # Extract noisy labels from loader's underlying dataset
    noisy_labels = bundle.noisy_eval_loader.dataset.tensors[1].numpy()
    return {
        "dataset": bundle.metadata.get("filename", ""),
        "topology": bundle.metadata.get("topology", ""),
        "noise_type": bundle.metadata.get("noise_type", ""),
        "noise_scale": bundle.metadata.get("noise_scale", 0),
        "n_samples": bundle.metadata.get("n_samples", 0),
        "d": bundle.d,
        "k": bundle.k,
        "mlp": _model_metrics(
            train_result.mlp_evals, train_result.mlp_loss_curve, mlp, bundle.k,
            noisy_labels, bundle.y_clean, bundle.flip_mask,
        ),
        "rbf": _model_metrics(
            train_result.rbf_evals, train_result.rbf_loss_curve, rbf, bundle.k,
            noisy_labels, bundle.y_clean, bundle.flip_mask,
        ),
    }


def save_metrics(metrics: dict, config: PipelineConfig) -> Path:
    """Save metrics JSON. Returns path."""
    out_dir = config.metrics_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{metrics['topology']}_{metrics['noise_type']}_{metrics['noise_scale']}.json"
    path = out_dir / fname
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path


def load_all_metrics(output_dir: str) -> list[dict]:
    """Load all metric JSONs from outputs/metrics/. For aggregated plots."""
    paths = sorted(glob(str(Path(output_dir) / "metrics" / "*.json")))
    results = []
    for p in paths:
        with open(p) as f:
            results.append(json.load(f))
    return results
