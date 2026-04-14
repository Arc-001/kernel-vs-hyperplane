"""Automated sweep over all .npz datasets in data/generated/."""

import logging
import sys
from glob import glob
from pathlib import Path

from config import PipelineConfig, get_config
from runners.run_single import run_single
from src.metrics import load_all_metrics
from src.viz import plot_degradation_curves, plot_entropy_bars, plot_flip_curves

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def run_sweep(config: PipelineConfig):
    """Run pipeline on every .npz, then generate aggregated plots."""
    dataset_files = sorted(glob(str(Path(config.data_dir) / "*.npz")))
    if not dataset_files:
        logger.warning("No .npz files in %s", config.data_dir)
        return

    logger.info("Found %d datasets", len(dataset_files))

    for npz_path in dataset_files:
        logger.info(">>> Processing: %s", npz_path)
        run_single(npz_path, config)

    # Aggregated plots — reload from JSON (crash-safe)
    all_results = load_all_metrics(config.output_dir)
    if all_results:
        logger.info("Generating aggregated plots from %d results", len(all_results))
        plot_degradation_curves(all_results, config)
        plot_entropy_bars(all_results, config)
        plot_flip_curves(all_results, config)
    else:
        logger.warning("No metric JSONs found — skipping aggregated plots")

    logger.info("Sweep complete")


if __name__ == "__main__":
    run_sweep(get_config())
