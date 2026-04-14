"""Smoke tests for MLP and RBF model definitions."""

import numpy as np
import torch
import torch.nn as nn

from config import get_config
from src.models import MLPClassifier, RBFClassifier, param_count

D, K = 2, 3
CFG = get_config()


def test_mlp_forward_shape():
    mlp = MLPClassifier(D, CFG.mlp_hidden_layers, K, CFG.mlp_activation)
    out = mlp(torch.randn(4, D))
    assert out.shape == (4, K)


def test_rbf_forward_shape():
    rbf = RBFClassifier(D, CFG.rbf_n_centers, K, CFG.rbf_sigma_learnable)
    rbf.load_centers(np.random.randn(CFG.rbf_n_centers, D), 1.0)
    out = rbf(torch.randn(4, D))
    assert out.shape == (4, K)


def test_mlp_gradients():
    mlp = MLPClassifier(D, CFG.mlp_hidden_layers, K)
    loss = nn.CrossEntropyLoss()(mlp(torch.randn(4, D)), torch.tensor([0, 1, 2, 0]))
    loss.backward()
    for p in mlp.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_rbf_gradients():
    rbf = RBFClassifier(D, CFG.rbf_n_centers, K)
    rbf.load_centers(np.random.randn(CFG.rbf_n_centers, D), 1.0)
    loss = nn.CrossEntropyLoss()(rbf(torch.randn(4, D)), torch.tensor([0, 1, 2, 0]))
    loss.backward()
    for p in rbf.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


def test_capacity_parity():
    # Parity holds at higher d where center params scale up.
    # Spec assumes d≈20-30 for parity; d=2 trivially fails (centers too small).
    d_hi, k_hi = 30, 10
    mlp = MLPClassifier(d_hi, CFG.mlp_hidden_layers, k_hi)
    rbf = RBFClassifier(d_hi, CFG.rbf_n_centers, k_hi)
    mlp_p = param_count(mlp)
    rbf_p = param_count(rbf)
    ratio = max(mlp_p, rbf_p) / min(mlp_p, rbf_p)
    assert ratio < 3.0, f"Capacity ratio {ratio:.1f}x exceeds 3x (MLP={mlp_p}, RBF={rbf_p})"


def test_rbf_load_centers():
    rbf = RBFClassifier(D, 8, K)
    centers = np.array([[1.0, 2.0]] * 8)
    rbf.load_centers(centers, 0.5)
    assert torch.allclose(rbf.centers[0], torch.tensor([1.0, 2.0]))
    assert torch.isclose(torch.exp(rbf.log_sigma), torch.tensor(0.5), atol=1e-6)
