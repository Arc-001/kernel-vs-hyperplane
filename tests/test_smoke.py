"""Phase 7 — Consolidated validation and sanity checks.

Six checks run before launching a full sweep:
1. Shape: d and k match metadata after ingestion
2. Forward pass: both models produce (batch, k) output
3. Gradient: one training step yields finite non-None grads
4. Capacity: param counts within 3x of each other (at d>=30)
5. Scaler leakage: scaler.mean_ derived from X_noisy, not X_clean
6. Memory: teardown returns GPU alloc to baseline
"""

import gc
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from config import get_config
from src.ingest import ingest
from src.models import MLPClassifier, RBFClassifier, param_count
from src.engine import init_rbf_centers, train, cleanup

FIXTURE_PATH = "data/generated/test_fixture.npz"
CFG = get_config(batch_size=32, epochs=3, rbf_n_centers=16)


@pytest.fixture(scope="module", autouse=True)
def ensure_fixture():
    if not Path(FIXTURE_PATH).exists():
        from tests.make_fixture import make_fixture
        make_fixture(FIXTURE_PATH)


@pytest.fixture(scope="module")
def bundle():
    return ingest(FIXTURE_PATH, CFG)


# --- 1. Shape test ---

def test_shape_matches_metadata(bundle):
    """d and k from ingestion match raw .npz metadata."""
    data = np.load(FIXTURE_PATH, allow_pickle=True)
    meta = data["metadata"].item()
    assert bundle.d == data["X_noisy"].shape[1]
    assert bundle.k == meta["n_classes"]


# --- 2. Forward pass test ---

def test_forward_pass_shapes(bundle):
    """Both models produce (batch, k) logits from (batch, d) input."""
    batch = 4
    x = torch.randn(batch, bundle.d)

    mlp = MLPClassifier(bundle.d, CFG.mlp_hidden_layers, bundle.k, CFG.mlp_activation)
    rbf = RBFClassifier(bundle.d, CFG.rbf_n_centers, bundle.k, CFG.rbf_sigma_learnable)

    assert mlp(x).shape == (batch, bundle.k)
    assert rbf(x).shape == (batch, bundle.k)


# --- 3. Gradient test ---

def test_gradients_finite(bundle):
    """One training step → all .grad non-None and finite."""
    mlp = MLPClassifier(bundle.d, CFG.mlp_hidden_layers, bundle.k, CFG.mlp_activation)
    rbf = RBFClassifier(bundle.d, CFG.rbf_n_centers, bundle.k, CFG.rbf_sigma_learnable)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(8, bundle.d)
    y = torch.randint(0, bundle.k, (8,))

    for model in (mlp, rbf):
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()

        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"{name} grad is None"
                assert torch.isfinite(p.grad).all(), f"{name} grad not finite"


# --- 4. Capacity parity test (d=30, k=10) ---

def test_capacity_parity():
    """At d=30, k=10 with default 256 centers, param counts within 3x."""
    d, k = 30, 10
    default_cfg = get_config()
    mlp = MLPClassifier(d, default_cfg.mlp_hidden_layers, k, default_cfg.mlp_activation)
    rbf = RBFClassifier(d, default_cfg.rbf_n_centers, k, default_cfg.rbf_sigma_learnable)

    mlp_p = param_count(mlp)
    rbf_p = param_count(rbf)
    ratio = max(mlp_p, rbf_p) / min(mlp_p, rbf_p)
    assert ratio <= 3.0, f"Capacity ratio {ratio:.1f}x exceeds 3x (MLP={mlp_p}, RBF={rbf_p})"


# --- 5. Scaler leakage test ---

def test_scaler_fit_on_noisy_only(bundle):
    """Scaler mean derived from X_noisy, not X_clean."""
    data = np.load(FIXTURE_PATH, allow_pickle=True)
    X_noisy = data["X_noisy"]
    X_clean = data["X_clean"]

    # Scaler should match noisy stats
    np.testing.assert_allclose(bundle.scaler.mean_, X_noisy.mean(axis=0), atol=1e-6)
    # And differ from clean stats (unless identical by coincidence — check not equal)
    if not np.allclose(X_noisy.mean(axis=0), X_clean.mean(axis=0), atol=1e-6):
        assert not np.allclose(bundle.scaler.mean_, X_clean.mean(axis=0), atol=1e-6)


# --- 6. Memory test (GPU only) ---

@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_memory_returns_to_baseline(bundle):
    """Train 3 epochs → cleanup → GPU alloc returns to baseline."""
    torch.cuda.empty_cache()
    gc.collect()
    baseline = torch.cuda.memory_allocated()

    n_centers = min(CFG.rbf_n_centers, len(bundle.X_noisy_scaled))
    mlp = MLPClassifier(bundle.d, CFG.mlp_hidden_layers, bundle.k, CFG.mlp_activation)
    rbf = RBFClassifier(bundle.d, n_centers, bundle.k, CFG.rbf_sigma_learnable)
    init_rbf_centers(rbf, bundle.X_noisy_scaled, CFG)
    train(mlp, rbf, bundle, CFG)

    cleanup(mlp, rbf)

    after = torch.cuda.memory_allocated()
    assert after <= baseline + 1024, f"Leaked {after - baseline} bytes after cleanup"
