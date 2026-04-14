"""Generate a small synthetic .npz test fixture matching the expected schema."""

import numpy as np
from pathlib import Path


def make_fixture(out_path: str = "data/generated/test_fixture.npz", n: int = 200, seed: int = 42):
    rng = np.random.RandomState(seed)

    # 3-class blobs in 2D
    centers = np.array([[0.0, 0.0], [3.0, 0.0], [1.5, 2.6]])
    labels = np.repeat([0, 1, 2], [n // 3, n // 3, n - 2 * (n // 3)])
    X_clean = centers[labels] + rng.randn(n, 2) * 0.4
    y_clean = labels.copy()

    # Noisy version: shift features + flip some labels
    X_noisy = X_clean + rng.randn(n, 2) * 0.3

    flip_mask = np.zeros(n, dtype=bool)
    flip_idx = rng.choice(n, size=n // 10, replace=False)
    flip_mask[flip_idx] = True
    y_noisy = y_clean.copy()
    y_noisy[flip_idx] = (y_noisy[flip_idx] + 1) % 3

    metadata = {
        "topology": "blobs",
        "noise_type": "gaussian",
        "noise_scale": 0.3,
        "n_samples": n,
        "n_classes": 3,
        "filename": "test_fixture.npz",
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        X_noisy=X_noisy,
        y_noisy=y_noisy,
        X_clean=X_clean,
        y_clean=y_clean,
        metadata=metadata,
        flip_mask=flip_mask,
    )
    print(f"Fixture saved: {out_path} ({n} samples, 2D, 3 classes)")


if __name__ == "__main__":
    make_fixture()
