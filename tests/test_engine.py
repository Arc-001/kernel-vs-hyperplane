"""Smoke tests for training engine."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from config import get_config
from src.ingest import ingest
from src.models import MLPClassifier, RBFClassifier, param_count
from src.engine import init_rbf_centers, train, evaluate, cleanup

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


def test_init_rbf_centers(bundle):
    rbf = RBFClassifier(bundle.d, CFG.rbf_n_centers, bundle.k)
    init_rbf_centers(rbf, bundle.X_noisy_scaled, CFG)
    assert not torch.all(rbf.centers == 0), "Centers should be non-zero after init"
    sigma = torch.exp(rbf.log_sigma).item()
    assert sigma > 0
    assert math.isfinite(sigma)


def test_train_runs(bundle):
    mlp = MLPClassifier(bundle.d, CFG.mlp_hidden_layers, bundle.k, CFG.mlp_activation)
    rbf = RBFClassifier(bundle.d, CFG.rbf_n_centers, bundle.k, CFG.rbf_sigma_learnable)
    init_rbf_centers(rbf, bundle.X_noisy_scaled, CFG)

    result = train(mlp, rbf, bundle, CFG)

    assert len(result.mlp_loss_curve) == 3
    assert len(result.rbf_loss_curve) == 3
    assert all(math.isfinite(l) for l in result.mlp_loss_curve)
    assert all(math.isfinite(l) for l in result.rbf_loss_curve)


def test_evaluate_shapes(bundle):
    mlp = MLPClassifier(bundle.d, CFG.mlp_hidden_layers, bundle.k, CFG.mlp_activation)
    mlp.to(CFG.device)
    ev = evaluate(mlp, bundle.clean_eval_loader, CFG.device)

    assert ev.logits.shape == (200, bundle.k)
    assert 0.0 <= ev.accuracy <= 1.0
    assert math.isfinite(ev.loss)
    assert ev.logits.device == torch.device("cpu")


def test_eval_results_in_train(bundle):
    mlp = MLPClassifier(bundle.d, CFG.mlp_hidden_layers, bundle.k, CFG.mlp_activation)
    rbf = RBFClassifier(bundle.d, CFG.rbf_n_centers, bundle.k, CFG.rbf_sigma_learnable)
    init_rbf_centers(rbf, bundle.X_noisy_scaled, CFG)

    result = train(mlp, rbf, bundle, CFG)

    for key in ["noisy", "clean"]:
        assert key in result.mlp_evals
        assert key in result.rbf_evals
        assert result.mlp_evals[key].logits.shape[0] == 200
        assert result.rbf_evals[key].logits.shape[0] == 200
