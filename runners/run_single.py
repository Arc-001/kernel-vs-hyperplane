"""Run train+eval pipeline on a single .npz dataset."""

import logging
import sys

from config import PipelineConfig, get_config
from src.ingest import ingest
from src.models import MLPClassifier, RBFClassifier, param_count
from src.engine import init_rbf_centers, train, cleanup
from src.metrics import compute_metrics, save_metrics
from src.viz import plot_boundaries

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def run_single(npz_path: str, config: PipelineConfig) -> dict | None:
    """Full pipeline for one dataset. Returns metrics dict or None on failure."""
    mlp = rbf = None
    try:
        bundle = ingest(npz_path, config)
        logger.info("Loaded %s — d=%d k=%d n=%d", npz_path, bundle.d, bundle.k, bundle.metadata.get("n_samples", 0))

        n_centers = min(config.rbf_n_centers, len(bundle.X_noisy_scaled))
        mlp = MLPClassifier(bundle.d, config.mlp_hidden_layers, bundle.k, config.mlp_activation)
        rbf = RBFClassifier(bundle.d, n_centers, bundle.k, config.rbf_sigma_learnable)
        logger.info("Params — MLP=%d  RBF=%d", param_count(mlp), param_count(rbf))

        init_rbf_centers(rbf, bundle.X_noisy_scaled, config)
        result = train(mlp, rbf, bundle, config)

        metrics = compute_metrics(result, bundle, mlp, rbf, config)
        path = save_metrics(metrics, config)
        logger.info("Metrics saved: %s", path)

        plot_boundaries(mlp, rbf, bundle, config)

        return metrics

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("OOM on %s — skipping", npz_path)
        else:
            logger.exception("Error on %s", npz_path)
        return None
    except Exception:
        logger.exception("Error on %s", npz_path)
        return None
    finally:
        if mlp is not None or rbf is not None:
            cleanup(mlp, rbf)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m runners.run_single <path.npz>")
        sys.exit(1)
    run_single(sys.argv[1], get_config())
