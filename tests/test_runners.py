"""Smoke tests for runners."""

from pathlib import Path

import pytest

from config import get_config
from runners.run_single import run_single
from runners.run_sweep import run_sweep

FIXTURE_PATH = "data/generated/test_fixture.npz"


@pytest.fixture(scope="module", autouse=True)
def ensure_fixture():
    if not Path(FIXTURE_PATH).exists():
        from tests.make_fixture import make_fixture
        make_fixture(FIXTURE_PATH)


def test_run_single(tmp_path):
    cfg = get_config(batch_size=32, epochs=3, rbf_n_centers=16, output_dir=str(tmp_path))
    metrics = run_single(FIXTURE_PATH, cfg)
    assert metrics is not None
    assert metrics["topology"] == "blobs"
    assert (tmp_path / "metrics").exists()
    assert len(list((tmp_path / "metrics").glob("*.json"))) == 1
    assert len(list((tmp_path / "figures").glob("boundary_*.png"))) == 1


def test_run_sweep(tmp_path):
    cfg = get_config(
        batch_size=32, epochs=3, rbf_n_centers=16,
        data_dir="data/generated", output_dir=str(tmp_path),
    )
    run_sweep(cfg)
    assert len(list((tmp_path / "metrics").glob("*.json"))) >= 1
    # Aggregated plots generated (at least entropy bars per result)
    assert len(list((tmp_path / "figures").glob("*.png"))) >= 1


def test_run_single_bad_path(tmp_path):
    cfg = get_config(output_dir=str(tmp_path))
    result = run_single("nonexistent.npz", cfg)
    assert result is None
