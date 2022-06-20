import logging
from typing import List, Iterable

from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
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
    logger.info("Getting the fingerprints model.")
    model, tokenizer = get_model_and_tokenizer(model_path, force_no_cuda)
    return RXNBERTFingerprintGenerator(model, tokenizer)


def generate_fps(model: str,
                 reaction_smiles: Iterable[str],
                 verbose: bool = False) -> List[List[float]]:
    rxnfp_generator = get_fingerprint_generator(model)

    if verbose:
        reaction_smiles = (smiles for smiles in tqdm(reaction_smiles))
    fps = [rxnfp_generator.convert(rxn) for rxn in reaction_smiles]
    logger.info(f'Generated {len(fps)} fps')
    return fps
