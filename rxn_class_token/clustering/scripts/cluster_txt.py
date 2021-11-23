import logging
import pickle
from typing import Tuple, Iterable

import click
import numpy as np
from rxn_chemutils.reaction_equation import sort_compounds
from rxn_chemutils.reaction_smiles import parse_any_reaction_smiles
from rxn_chemutils.tokenization import detokenize_smiles
from rxn_utilities.file_utilities import is_path_creatable, count_lines, iterate_lines_from_file

from ..clusterer import Clusterer
from ..fingerprints import get_fingerprint_generator, generate_fps_in_chunks

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def process_reaction(reaction_smiles: str) -> str:
    # NB: the sub-fragments will not be reordered. They will stay together.
    reaction_smiles = detokenize_smiles(reaction_smiles)
    reaction = parse_any_reaction_smiles(reaction_smiles)
    reaction = sort_compounds(reaction)
    return reaction.to_string('.')


@click.command()
@click.option(
    '--input_txt',
    '-i',
    multiple=True,
    help='If 1 file: TXT with the reactions; '
    'if 2 files: 1 TXT for the reactants and 1 TXT for the products. '
    'Everything is assumed to be canonicalized.'
)
@click.option('--output_txt', '-o', type=str, required=True)
@click.option('--fp_model_path', '-f', type=str, required=True, help='Path to the fp model.')
@click.option('--clusterer_pkl', '-c', type=str, required=True, help='Path to the clusterer.')
@click.option('--chunk_size', '-s', type=int, default=512, help='Chunk size.')
def main(
    input_txt: Tuple[str, ...], output_txt: str, fp_model_path: str, clusterer_pkl: str,
    chunk_size: int
):
    """Get the cluster ids of reactions via their fingerprints."""

    if not is_path_creatable(output_txt):
        raise ValueError(f'Permissions insufficient to create file "{output_txt}".')

    with open(clusterer_pkl, 'rb') as f_clusterer:
        clusterer: Clusterer = pickle.load(f_clusterer)

    reactions_raw: Iterable[str]
    if len(input_txt) == 1:
        reactions_txt = input_txt[0]
        number_reactions = count_lines(reactions_txt)
        logger.info(f'Will get clusters for {number_reactions} reactions from "{reactions_txt}".')
        reactions_raw = iterate_lines_from_file(reactions_txt)
    elif len(input_txt) == 2:
        precursors_txt = input_txt[0]
        products_txt = input_txt[1]
        number_precursors = count_lines(precursors_txt)
        number_products = count_lines(products_txt)
        if number_precursors != number_products:
            raise RuntimeError(
                f'"{precursors_txt}" and "{products_txt}" have a different number '
                f'of lines: {number_precursors} vs {number_products}'
            )
        logger.info(
            f'Will get clusters for {number_precursors} reactions '
            f'from "{precursors_txt}" and "{products_txt}".'
        )
        precursors = iterate_lines_from_file(precursors_txt)
        products = iterate_lines_from_file(products_txt)
        reactions_raw = (
            f'{precursor} >> {product}' for precursor, product in zip(precursors, products)
        )
    else:
        raise ValueError('Invalid number of input files')

    reaction_smiles = (process_reaction(rxn) for rxn in reactions_raw)

    fp_generator = get_fingerprint_generator(fp_model_path)

    with open(output_txt, 'wt') as f:

        # Generate the fingerprints, chunk by chunk, and append the results to the output file
        for chunk_fps in generate_fps_in_chunks(fp_generator, reaction_smiles, chunk_size):
            chunk_clusters = clusterer.predict(np.array(chunk_fps))

            for cluster in chunk_clusters:
                f.write(f'{str(cluster)}\n')


if __name__ == '__main__':
    main()
