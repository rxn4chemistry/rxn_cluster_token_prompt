import logging
from typing import Iterable, List

from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_model_and_tokenizer(force_no_cuda: bool = False):
    import torch
    from rxnfp.transformer_fingerprints import get_default_model_and_tokenizer

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and not force_no_cuda) else "cpu"
    )

    model, tokenizer = get_default_model_and_tokenizer()
    model = model.eval()
    model.to(device)
    return model, tokenizer


def get_fingerprint_generator(force_no_cuda: bool = False):
    from rxnfp.transformer_fingerprints import RXNBERTFingerprintGenerator

    logger.info("Getting the fingerprints model.")
    model, tokenizer = get_model_and_tokenizer(force_no_cuda)
    return RXNBERTFingerprintGenerator(model, tokenizer)


def generate_fps(
    reaction_smiles: Iterable[str], verbose: bool = False
) -> List[List[float]]:
    rxnfp_generator = get_fingerprint_generator()

    if verbose:
        reaction_smiles = (smiles for smiles in tqdm(reaction_smiles))
    fps = [rxnfp_generator.convert(rxn) for rxn in reaction_smiles]
    logger.info(f"Generated {len(fps)} fps")
    return fps
