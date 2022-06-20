import logging
from typing import Optional

from rxn.utilities.files import PathLike, is_path_exists_or_creatable

from rxn_cluster_token_prompt.onmt_utils.utils import detokenize_file, file_is_tokenized, tokenize_file
from rxn_cluster_token_prompt.onmt_utils.translate import translate

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def forward_or_retro_translation(
    src_file: PathLike,
    tgt_file: Optional[PathLike],
    pred_file: PathLike,
    model: PathLike,
    n_best: int,
    beam_size: int,
    batch_size: int,
    gpu: bool,
    max_length: int = 300,
    as_external_command: bool = False,
) -> None:
    """
    Do a forward or retro translation.
    This function takes care of tokenizing/detokenizing the input. In principle, by adapting
    the "invalid" placeholder, this could also work when input/output are full reactions.
    Note: no check is made that the source is canonical.
    Args:
        src_file: source file (tokenized or detokenized).
        tgt_file: ground truth file (tokenized or detokenized), not mandatory.
        pred_file: file where to save the predictions.
        model: model to do the translation
        n_best: number of predictions to make for each input.
        beam_size: beam size.
        batch_size: batch size.
        gpu: whether to use the GPU.
        max_length: maximum sequence length.
        as_external_command: runs the onmt command instead of Python code.
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
    elif file_is_tokenized(tgt_file):
        tokenized_tgt = tgt_file
    else:
        tokenized_tgt = str(tgt_file) + ".tokenized"
        tokenize_file(tgt_file, tokenized_tgt, invalid_placeholder="")

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

    detokenize_file(tokenized_pred, pred_file)