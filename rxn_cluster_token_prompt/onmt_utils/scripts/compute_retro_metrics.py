import json
import logging
from pathlib import Path

import click
from rxn.utilities.logging import setup_console_logger

from rxn_cluster_token_prompt.onmt_utils.retro_metrics import RetroMetrics
from rxn_cluster_token_prompt.onmt_utils.utils import RetroFiles

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@click.command(context_settings={"show_default": True})
@click.option("--results_dir", required=True, help="Where the retro predictions are stored")
@click.option(
    "--reordered",
    required=False,
    is_flag=True,
    show_default=True,
    default=False,
    help="Whether to use the reordered files - for class token.",
)
def main(results_dir: str, reordered: bool) -> None:
    """Starting from the predictions files for retro, calculate the default metrics."""

    results_path = Path(results_dir)
    output_path_contains_files = any(results_path.iterdir())
    if not output_path_contains_files:
        raise RuntimeError(f'This directory "{results_path}" is empty.')

    retro_files = RetroFiles(results_path)

    # Setup logging (to terminal)
    setup_console_logger()

    logger.info("Computing the retro metrics...")
    metrics = RetroMetrics.from_retro_files(retro_files, reordered=reordered)
    metrics_dict = metrics.get_metrics()
    with open(retro_files.metrics_file, "wt") as f:
        json.dump(metrics_dict, f, indent=2)
    logger.info(f'Computing the retro metrics... Saved to "{retro_files.metrics_file}".')


if __name__ == "__main__":
    main()
