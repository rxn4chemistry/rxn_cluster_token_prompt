import logging
import os
import time
from pathlib import Path
from typing import List, Iterable, Generator

from rxn_utilities.container_utilities import chunker
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_model_and_tokenizer(model_path: str, force_no_cuda: bool = False):
    import torch
    from transformers import BertModel
    from rxnfp.tokenization import SmilesTokenizer

    tokenizer_vocab_path = model_path + '/vocab.txt'
    device = torch.device("cuda" if (torch.cuda.is_available() and not force_no_cuda) else "cpu")

    model = BertModel.from_pretrained(model_path)
    model = model.eval()
    model.to(device)

    tokenizer = SmilesTokenizer(tokenizer_vocab_path)
    return model, tokenizer


def get_fingerprint_generator(model_path: str, force_no_cuda: bool = False):
    from rxnfp.transformer_fingerprints import RXNBERTFingerprintGenerator
    model, tokenizer = get_model_and_tokenizer(model_path, force_no_cuda)
    return RXNBERTFingerprintGenerator(model, tokenizer)


def generate_fps_in_chunks(fingerprint_generator, reaction_smiles: Iterable[str],
                           chunk_size: int) -> Generator[List[float], None, None]:
    start = time.time()
    n_generated = 0
    for chunk_reactions in chunker(reaction_smiles, chunk_size, filter_out_none=True):
        fp_chunk = fingerprint_generator.convert_batch(chunk_reactions)
        n_generated += len(fp_chunk)
        yield fp_chunk
    logger.info(
        f'Generated {n_generated} fps in {int(time.time() - start)} s, in chunks of {chunk_size}.'
    )


def generate_fps(model: str,
                 reaction_smiles: Iterable[str],
                 verbose: bool = False) -> List[List[float]]:
    rxnfp_generator = get_fingerprint_generator(model)

    if verbose:
        reaction_smiles = (smiles for smiles in tqdm(reaction_smiles))
    fps = [rxnfp_generator.convert(rxn) for rxn in reaction_smiles]
    logger.info(f'Generated {len(fps)} fps')
    return fps


def generate_pistachio_fps(reaction_smiles: Iterable[str],
                           verbose: bool = False) -> List[List[float]]:
    xxx_fp_dir = Path(os.environ['XXX_FP_DIR'])
    pistachio_model = xxx_fp_dir / 'fp_model_pistachio'
    return generate_fps(str(pistachio_model), reaction_smiles, verbose)


def generate_1k_tpl_fps(reaction_smiles: Iterable[str],
                        verbose: bool = False) -> List[List[float]]:
    xxx_fp_dir = Path(os.environ['XXX_FP_DIR'])
    tpl_1k_model = xxx_fp_dir / 'fp_model_1k_tpl'
    return generate_fps(str(tpl_1k_model), reaction_smiles, verbose)
