"""Smoke tests for data ingestion pipeline."""

import numpy as np
import pytest
import torch
from pathlib import Path

from config import get_config
from src.ingest import ingest

FIXTURE_PATH = "data/generated/test_fixture.npz"


@pytest.fixture(scope="module", autouse=True)
def ensure_fixture():
    if not Path(FIXTURE_PATH).exists():
        from tests.make_fixture import make_fixture
        make_fixture(FIXTURE_PATH)


@pytest.fixture(scope="module")
def bundle():
    return ingest(FIXTURE_PATH, get_config(batch_size=32))


def test_dimensions(bundle):
    assert bundle.d == 2
    assert bundle.k == 3


def test_scaler_fitted_on_noisy(bundle):
    assert hasattr(bundle.scaler, "mean_")
    assert bundle.scaler.mean_.shape == (bundle.d,)


def test_loader_batch_shapes(bundle):
    X_batch, y_batch = next(iter(bundle.train_loader))
    assert X_batch.shape == (32, 2)
    assert y_batch.shape == (32,)
    assert X_batch.dtype == torch.float32
    assert y_batch.dtype == torch.long


def test_raw_arrays_present(bundle):
    assert isinstance(bundle.X_noisy_scaled, np.ndarray)
    assert isinstance(bundle.X_clean_scaled, np.ndarray)
    assert bundle.X_noisy_scaled.shape == (200, 2)
    assert bundle.X_clean_scaled.shape == (200, 2)


def test_flip_mask(bundle):
    assert bundle.flip_mask is not None
    assert len(bundle.flip_mask) == 200
    assert bundle.flip_mask.sum() == 20  # 10% of 200


def test_metadata_keys(bundle):
    for key in ["topology", "noise_type", "noise_scale", "n_samples"]:
        assert key in bundle.metadata


def test_y_clean_preserved(bundle):
    assert isinstance(bundle.y_clean, np.ndarray)
    assert len(bundle.y_clean) == 200


def test_tensors_on_cpu(bundle):
    X_batch, _ = next(iter(bundle.train_loader))
    assert X_batch.device == torch.device("cpu")
