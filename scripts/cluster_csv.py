import pickle
from typing import List

import click
import pandas as pd
from rxn_utilities.file_utilities import is_path_creatable

from ..clusterer import Clusterer


@click.command()
@click.option('--input_csv', '-i', type=str, required=True)
@click.option('--output_csv', '-o', type=str, required=True)
@click.option('--clusterer_pkl', '-p', type=str, required=True, help='Path to the clusterer.')
@click.option(
    '--rxn_smiles_column',
    '-r',
    default='rxn_smiles',
    required=True,
    help='Column to get the reaction SMILES from.'
)
@click.option(
    '--cluster_column',
    '-m',
    default='cluster_id',
    required=True,
    help='Column to write the cluster id to.'
)
def main(
    input_csv: str, output_csv: str, clusterer_pkl: str, rxn_smiles_column: str,
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

    # Function to use below in "assign", basically generates the new desired
    # column from the full DataFrame. We do not use "apply" because it would
    # get the cluster id line by line, which is not efficient.
    def to_cluster_array(full_df: pd.DataFrame) -> List[int]:
        smiles = full_df[rxn_smiles_column].tolist()
        return clusterer.get_cluster_nos(smiles, verbose=True)

    df = df.assign(**{cluster_column: to_cluster_array})

    # Save as CSV
    df.to_csv(output_csv, index=False)


if __name__ == '__main__':
    main()
