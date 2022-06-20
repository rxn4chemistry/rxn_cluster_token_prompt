import logging
from pathlib import Path
from typing import Optional, Union

from rxn_utilities.file_utilities import is_path_exists_or_creatable

from rxn_cluster_token_prompt.onmt_utils.utils import (
    classification_file_is_tokenized,
    detokenize_classification_file,
    file_is_tokenized,
    tokenize_classification_file,
    tokenize_file,
)
from rxn_cluster_token_prompt.onmt_utils.translate import translate

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def classification_translation(
    src_file: Union[str, Path],
    tgt_file: Optional[Union[str, Path]],
    pred_file: Union[str, Path],
    model: Union[str, Path],
    n_best: int,
    beam_size: int,
    batch_size: int,
    gpu: bool,
    max_length: int = 3,
    as_external_command: bool = False,
) -> None:
    """
    Do a classification translation.
    This function takes care of tokenizing/detokenizing the input.
    Note: no check is made that the source is canonical.
    Args:
        src_file: source file (tokenized or detokenized).
        tgt_file: ground truth class file (tokenized), not mandatory.
        pred_file: file where to save the predictions.
        model: model to do the translation
        n_best: number of predictions to make for each input.
        beam_size: beam size.
        batch_size: batch size.
        gpu: whether to use the GPU.
        max_length: maximum sequence length.
    """
    if not is_path_exists_or_creatable(pred_file):
        raise RuntimeError(f'The file "{pred_file}" cannot be created.')

    # src
    if file_is_tokenized(src_file):
        tokenized_src = src_file
    else:
        tokenized_src = str(src_file) + ".tokenized"
        tokenize_file(src_file, tokenized_src, invalid_placeholder="")

    # tgt
    if tgt_file is None:
        tokenized_tgt = None
    elif classification_file_is_tokenized(tgt_file):
        tokenized_tgt = tgt_file
    else:
        tokenized_tgt = str(tgt_file) + ".tokenized"
        tokenize_classification_file(tgt_file, tokenized_tgt)

    tokenized_pred = str(pred_file) + ".tokenized"

    translate(
        model=model,
        src=tokenized_src,
        tgt=tokenized_tgt,
        output=tokenized_pred,
        n_best=n_best,
        beam_size=beam_size,
        max_length=max_length,
        batch_size=batch_size,
        gpu=gpu,
        as_external_command=as_external_command,
    )

    detokenize_classification_file(tokenized_pred, pred_file)
