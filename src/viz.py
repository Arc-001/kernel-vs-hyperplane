"""Phase 5 — Visualization engine."""

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import PipelineConfig
from src.ingest import DatasetBundle


def _predict_grid(model, grid: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    """Batched no_grad inference on mesh grid. Returns class predictions."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(grid), batch_size):
            batch = torch.tensor(grid[i:i + batch_size], dtype=torch.float32).to(device)
            logits = model(batch)
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def plot_boundaries(mlp, rbf, bundle: DatasetBundle, config: PipelineConfig):
    """5A — 2D decision boundaries. Skips if d != 2."""
    if bundle.d != 2:
        return

    X = bundle.X_noisy_scaled
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, config.mesh_resolution),
        np.linspace(y_min, y_max, config.mesh_resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    cmap = plt.cm.Set2
    scatter_colors = bundle.y_clean

    for ax, model, title in [(axes[0], mlp, "MLP"), (axes[1], rbf, "RBF")]:
        preds = _predict_grid(model, grid, config.device, config.inference_batch_size)
        preds = preds.reshape(xx.shape)
        ax.contourf(xx, yy, preds, alpha=0.3, cmap=cmap)
        ax.scatter(
            bundle.X_clean_scaled[:, 0], bundle.X_clean_scaled[:, 1],
            c=scatter_colors, cmap=cmap, s=8, edgecolors="k", linewidths=0.3,
        )
        ax.set_title(title)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    meta = bundle.metadata
    fig.suptitle(f"{meta.get('topology', '')} | {meta.get('noise_type', '')} | scale={meta.get('noise_scale', '')}")
    fig.tight_layout()

    out_dir = config.figures_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"boundary_{meta.get('topology', '')}_{meta.get('noise_type', '')}_{meta.get('noise_scale', '')}.png"
    fig.savefig(out_dir / fname, dpi=150)
    plt.close(fig)


def plot_degradation_curves(all_results: list[dict], config: PipelineConfig):
    """5B — Accuracy vs noise_scale, grouped by (topology, noise_type)."""
    groups = defaultdict(list)
    for r in all_results:
        groups[(r["topology"], r["noise_type"])].append(r)

    out_dir = config.figures_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for (topo, noise), items in groups.items():
        items.sort(key=lambda x: x["noise_scale"])
        scales = [r["noise_scale"] for r in items]
        mlp_acc = [r["mlp"]["clean_acc"] for r in items]
        rbf_acc = [r["rbf"]["clean_acc"] for r in items]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(scales, mlp_acc, "o-", label="MLP")
        ax.plot(scales, rbf_acc, "s--", label="RBF")
        ax.set_xlabel("Noise Scale")
        ax.set_ylabel("Clean Accuracy")
        ax.set_title(f"Degradation: {topo} / {noise}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"degradation_{topo}_{noise}.png", dpi=150)
        plt.close(fig)


def plot_entropy_bars(all_results: list[dict], config: PipelineConfig):
    """5C — Entropy bar chart per (topology, noise_scale)."""
    out_dir = config.figures_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for r in all_results:
        topo = r["topology"]
        noise = r["noise_type"]
        scale = r["noise_scale"]

        labels = ["Noisy Eval", "Clean Eval"]
        mlp_vals = [r["mlp"]["normalized_entropy_noisy"], r["mlp"]["normalized_entropy_clean"]]
        rbf_vals = [r["rbf"]["normalized_entropy_noisy"], r["rbf"]["normalized_entropy_clean"]]

        x = np.arange(len(labels))
        w = 0.35
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - w / 2, mlp_vals, w, label="MLP")
        ax.bar(x + w / 2, rbf_vals, w, label="RBF")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Normalized Entropy")
        ax.set_title(f"Entropy: {topo} / {noise} / scale={scale}")
        ax.set_ylim(0, 1.1)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"entropy_{topo}_{noise}_{scale}.png", dpi=150)
        plt.close(fig)


def plot_flip_curves(all_results: list[dict], config: PipelineConfig):
    """5D — Flip memorization rate vs noise_scale."""
    groups = defaultdict(list)
    for r in all_results:
        if r["mlp"]["flip_memorization_rate"] is not None:
            groups[(r["topology"], r["noise_type"])].append(r)

    out_dir = config.figures_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for (topo, noise), items in groups.items():
        items.sort(key=lambda x: x["noise_scale"])
        scales = [r["noise_scale"] for r in items]
        mlp_flip = [r["mlp"]["flip_memorization_rate"] for r in items]
        rbf_flip = [r["rbf"]["flip_memorization_rate"] for r in items]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(scales, mlp_flip, "o-", label="MLP")
        ax.plot(scales, rbf_flip, "s--", label="RBF")
        ax.set_xlabel("Noise Scale")
        ax.set_ylabel("Flip Memorization Rate")
        ax.set_title(f"Label Corruption: {topo} / {noise}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"flip_{topo}_{noise}.png", dpi=150)
        plt.close(fig)
