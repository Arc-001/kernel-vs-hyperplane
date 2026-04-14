"""Smoke tests for metric computation."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from config import get_config
from src.ingest import ingest
from src.models import MLPClassifier, RBFClassifier
from src.engine import init_rbf_centers, train
from src.metrics import compute_entropy, compute_flip_metrics, compute_metrics, save_metrics, load_all_metrics

FIXTURE_PATH = "data/generated/test_fixture.npz"
CFG = get_config(batch_size=32, epochs=3, rbf_n_centers=16)


@pytest.fixture(scope="module", autouse=True)
def ensure_fixture():
    if not Path(FIXTURE_PATH).exists():
        from tests.make_fixture import make_fixture
        make_fixture(FIXTURE_PATH)


def test_entropy_uniform():
    k = 4
    logits = torch.zeros(100, k)  # uniform probs
    mean, norm = compute_entropy(logits, k)
    assert abs(norm - 1.0) < 0.01


def test_entropy_confident():
    k = 4
    logits = torch.zeros(100, k)
    logits[:, 0] = 100.0  # near one-hot
    mean, norm = compute_entropy(logits, k)
    assert norm < 0.01


def test_flip_metrics_known():
    logits = torch.tensor([[10.0, 0.0, 0.0]] * 10)  # all predict class 0
    noisy_labels = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    clean_labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    flip_mask = np.array([True, True, False, False, False, False, False, False, False, False])
    result = compute_flip_metrics(logits, noisy_labels, clean_labels, flip_mask)
    assert result["acc_on_flipped"] == 1.0  # preds=0, clean=0 → correct
    assert result["flip_memorization_rate"] == 0.0  # preds=0, noisy=1 → not memorized


def test_flip_metrics_none():
    assert compute_flip_metrics(torch.zeros(5, 3), np.zeros(5), np.zeros(5), None) is None


def test_compute_metrics_full():
    bundle = ingest(FIXTURE_PATH, CFG)
    mlp = MLPClassifier(bundle.d, CFG.mlp_hidden_layers, bundle.k, CFG.mlp_activation)
    rbf = RBFClassifier(bundle.d, CFG.rbf_n_centers, bundle.k, CFG.rbf_sigma_learnable)
    init_rbf_centers(rbf, bundle.X_noisy_scaled, CFG)
    result = train(mlp, rbf, bundle, CFG)
    metrics = compute_metrics(result, bundle, mlp, rbf, CFG)

    # Top-level keys
    for key in ["topology", "noise_type", "noise_scale", "d", "k", "mlp", "rbf"]:
        assert key in metrics

    # Per-model keys
    for model_key in ["mlp", "rbf"]:
        m = metrics[model_key]
        for k in ["noisy_acc", "clean_acc", "perf_delta", "noisy_loss", "clean_loss",
                   "mean_entropy_noisy", "normalized_entropy_noisy",
                   "flip_memorization_rate", "param_count", "train_loss_curve"]:
            assert k in m
        assert len(m["train_loss_curve"]) == 3


def test_save_load_roundtrip(tmp_path):
    cfg = get_config(output_dir=str(tmp_path))
    metrics = {
        "topology": "blobs", "noise_type": "gaussian", "noise_scale": 0.3,
        "d": 2, "k": 3, "n_samples": 200, "dataset": "test",
        "mlp": {"noisy_acc": 0.9}, "rbf": {"noisy_acc": 0.85},
    }
    path = save_metrics(metrics, cfg)
    assert path.exists()
    loaded = load_all_metrics(str(tmp_path))
    assert len(loaded) == 1
    assert loaded[0]["mlp"]["noisy_acc"] == 0.9
