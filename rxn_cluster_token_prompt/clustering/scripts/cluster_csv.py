import os
import logging
import pickle
import random
from pathlib import Path
from typing import List, Optional, Dict

import click
import pandas as pd
from rxn.utilities.files import is_path_creatable
from rxn.utilities.logging import setup_console_logger

from rxn_cluster_token_prompt.clustering.clusterer import Clusterer
from rxn_cluster_token_prompt.clustering.data_loading import ensure_fp, FP_COLUMN

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@click.command()
@click.option(
    '--input_csv', '-i', type=str, required=True, help='Path to the input reactions csv.'
)
@click.option(
    '--output_csv', '-o', type=str, required=True, help='Path to the output reactions csv.'
)
@click.option('--clusterer_pkl', '-p', type=str, required=False, help='Path to the clusterer.')
@click.option(
    '--n_clusters_random',
    '-n',
    type=int,
    required=False,
    help='Number of random groups for the reaction '
    'classes.'
)
@click.option(
    '--cluster_column',
    '-m',
    default='cluster_id',
    required=True,
    help="Column to write the cluster id to, default 'cluster_id'."
)
@click.option(
    '--class_column',
    '-c',
    default='class',
    required=False,
    help="Column where the reaction classes are stored, default 'class'."
)
def main(
    input_csv: str, output_csv: str, clusterer_pkl: Optional[str],
    n_clusters_random: Optional[int], cluster_column: str, class_column: str
):
    setup_console_logger()
    """Get the cluster number and add it as a new column to a CSV."""
    if clusterer_pkl is not None and n_clusters_random is not None:
        raise ValueError("Choose between '--clusterer_pkl' and '--n_clusters_random'.")
    if clusterer_pkl is None and n_clusters_random is None:
        raise ValueError(
            "Either a pickle file for the clusterer or the number of random groups for the reaction "
            "classes must be provided"
        )

    if not is_path_creatable(output_csv):
        raise ValueError(f'Permissions insufficient to create file "{output_csv}".')

    # Read CSV
    df: pd.DataFrame = pd.read_csv(input_csv)

    if clusterer_pkl is not None:
        with open(clusterer_pkl, 'rb') as f:
            clusterer: Clusterer = pickle.load(f)

        logger.info("Ensuring reaction fingerprints ...")
        ensure_fp(df, Path(os.environ['FPS_SAVE_PATH']))

        # Function to use below in "assign", basically generates the new desired
        # column from the full DataFrame. We do not use "apply" because it would
        # get the cluster id line by line, which is not efficient.
        def to_cluster_array(full_df: pd.DataFrame) -> List[int]:
            fingerprints = full_df[FP_COLUMN].tolist()
            return clusterer.predict(fingerprints)

        logger.info("Using clusterer to predict cluster id ...")
        df = df.assign(**{cluster_column: to_cluster_array})
        df.drop(labels=FP_COLUMN, axis=1, inplace=True)

    else:
        if class_column not in df.columns:
            raise KeyError(f"The column '{class_column}' was not found in the data.")

        unique_classes = sorted(list(set(df[class_column].values)))
        logger.info(f"Found {len(unique_classes)} unique reaction classes.")
        if len(unique_classes) < n_clusters_random:
            raise ValueError(
                "Choose a number of clusters smaller that the number of unique reaction classes."
            )
        elif len(unique_classes) == n_clusters_random:
            logger.info("Number of clusters equals the number of unique classes")
            inverted_clusters_map = {cl: i for i, cl in enumerate(unique_classes)}
            logger.info(f"Random clusters map: {inverted_clusters_map}")
            df[cluster_column] = df[class_column].map(inverted_clusters_map)
        else:
            logger.info(f"Grouping randomly in {n_clusters_random} clusters.")
            random.seed(42)
            clusters_map: Dict[str, List[str]] = {i: [] for i in range(n_clusters_random)}
            while unique_classes:
                for i in range(n_clusters_random):
                    if len(unique_classes) < 1:
                        break
                    index = random.randrange(
                        start=0,
                        stop=len(unique_classes),
                    )
                    removed_element = unique_classes.pop(index)
                    clusters_map[i].append(removed_element)

            inverted_clusters_map = {i: k for k, v in clusters_map.items() for i in v}
            logger.info(f"Random clusters map: {inverted_clusters_map}")
            df[cluster_column] = df[class_column].map(inverted_clusters_map)

    print(df.head())

    # Save as CSV
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved to: {output_csv}")


if __name__ == '__main__':
    main()
