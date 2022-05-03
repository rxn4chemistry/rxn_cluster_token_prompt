import os
import pickle
from pathlib import Path
from typing import List

import click
import pandas as pd
from rxn_utilities.file_utilities import is_path_creatable

from rxn_class_token.clustering.clusterer import Clusterer
from rxn_class_token.clustering.data_loading import ensure_fp, FP_COLUMN


@click.command()
@click.option('--input_csv', '-i', type=str, required=True)
@click.option('--output_csv', '-o', type=str, required=True)
@click.option('--clusterer_pkl', '-p', type=str, required=True, help='Path to the clusterer.')
@click.option(
    '--cluster_column',
    '-m',
    default='cluster_id',
    required=True,
    help='Column to write the cluster id to.'
)
def cluster_csv(
    input_csv: str, output_csv: str, clusterer_pkl: str,
    cluster_column: str
):
    """Get the cluster number and add it as a new column to a CSV.

    NB: the other script, cluster_txt, may be more efficient."""

    if not is_path_creatable(output_csv):
        raise ValueError(f'Permissions insufficient to create file "{output_csv}".')

    with open(clusterer_pkl, 'rb') as f:
        clusterer: Clusterer = pickle.load(f)

    # Read CSV
    df: pd.DataFrame = pd.read_csv(input_csv)
    ensure_fp(df, Path(os.environ['FPS_SAVE_PATH']))

    # Function to use below in "assign", basically generates the new desired
    # column from the full DataFrame. We do not use "apply" because it would
    # get the cluster id line by line, which is not efficient.
    def to_cluster_array(full_df: pd.DataFrame) -> List[int]:
        fingerprints = full_df[FP_COLUMN].tolist()
        return clusterer.predict(fingerprints)

    df = df.assign(**{cluster_column: to_cluster_array})

    # Save as CSV
    df.drop(labels=FP_COLUMN, axis=1, inplace=True)
    df.to_csv(output_csv, index=False)


if __name__ == '__main__':
    cluster_csv()
