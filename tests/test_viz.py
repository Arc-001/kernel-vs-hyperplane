"""Smoke tests for visualization engine."""

from pathlib import Path

import pytest

from config import get_config
from src.ingest import ingest
from src.models import MLPClassifier, RBFClassifier
from src.engine import init_rbf_centers, train
from src.metrics import compute_metrics
from src.viz import plot_boundaries, plot_degradation_curves, plot_entropy_bars, plot_flip_curves

FIXTURE_PATH = "data/generated/test_fixture.npz"
CFG = get_config(batch_size=32, epochs=3, rbf_n_centers=16)


@pytest.fixture(scope="module", autouse=True)
def ensure_fixture():
    if not Path(FIXTURE_PATH).exists():
        from tests.make_fixture import make_fixture
        make_fixture(FIXTURE_PATH)


@pytest.fixture(scope="module")
def trained():
    """Train once, return (mlp, rbf, bundle, metrics)."""
    bundle = ingest(FIXTURE_PATH, CFG)
    mlp = MLPClassifier(bundle.d, CFG.mlp_hidden_layers, bundle.k, CFG.mlp_activation)
    rbf = RBFClassifier(bundle.d, CFG.rbf_n_centers, bundle.k, CFG.rbf_sigma_learnable)
    init_rbf_centers(rbf, bundle.X_noisy_scaled, CFG)
    result = train(mlp, rbf, bundle, CFG)
    metrics = compute_metrics(result, bundle, mlp, rbf, CFG)
    return mlp, rbf, bundle, metrics


def test_boundary_plot_creates_file(trained, tmp_path):
    mlp, rbf, bundle, _ = trained
    cfg = get_config(batch_size=32, epochs=3, rbf_n_centers=16, output_dir=str(tmp_path))
    plot_boundaries(mlp, rbf, bundle, cfg)
    pngs = list((tmp_path / "figures").glob("boundary_*.png"))
    assert len(pngs) == 1


def test_boundary_skips_high_d(trained, tmp_path):
    mlp, rbf, bundle, _ = trained
    # Fake d != 2
    bundle_fake = bundle.__class__(**{**bundle.__dict__, "d": 5})
    cfg = get_config(output_dir=str(tmp_path))
    plot_boundaries(mlp, rbf, bundle_fake, cfg)
    figs = tmp_path / "figures"
    assert not figs.exists() or len(list(figs.glob("*.png"))) == 0


def test_degradation_curves(trained, tmp_path):
    _, _, _, metrics = trained
    # Simulate 3 noise scales
    all_results = []
    for scale in [0.1, 0.3, 0.5]:
        m = {**metrics, "noise_scale": scale}
        all_results.append(m)
    cfg = get_config(output_dir=str(tmp_path))
    plot_degradation_curves(all_results, cfg)
    pngs = list((tmp_path / "figures").glob("degradation_*.png"))
    assert len(pngs) == 1


def test_entropy_bars(trained, tmp_path):
    _, _, _, metrics = trained
    cfg = get_config(output_dir=str(tmp_path))
    plot_entropy_bars([metrics], cfg)
    pngs = list((tmp_path / "figures").glob("entropy_*.png"))
    assert len(pngs) == 1


def test_flip_curves(trained, tmp_path):
    _, _, _, metrics = trained
    all_results = []
    for scale in [0.1, 0.3]:
        all_results.append({**metrics, "noise_scale": scale})
    cfg = get_config(output_dir=str(tmp_path))
    plot_flip_curves(all_results, cfg)
    pngs = list((tmp_path / "figures").glob("flip_*.png"))
    assert len(pngs) == 1
