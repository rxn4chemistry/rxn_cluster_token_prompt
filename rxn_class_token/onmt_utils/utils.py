import logging

import numpy as np
from rxn.chemutils.conversion import canonicalize_smiles
from rxn.chemutils.exceptions import InvalidSmiles

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def maybe_canonicalize(smiles: str, check_valence: bool = True, invalid_replacement='') -> str:
    """
    Canonicalize a SMILES string, but returns the original SMILES string if it fails.
    """
    try:
        return canonicalize_smiles(smiles, check_valence=check_valence)
    except InvalidSmiles:
        return invalid_replacement


def compute_probabilities(log_probability: float):
    return np.exp(log_probability)


def create_rxn(precursors: str, product: str):
    return f"{precursors}>>{product}"


def convert_class_token_idx_for_tranlation_models(class_token_idx: int) -> str:
    return f"[{class_token_idx}]"
