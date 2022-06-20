import logging
from pathlib import Path
from typing import List, Tuple, Union

import click
from rxn.utilities.containers import chunker
from rxn.utilities.files import dump_list_to_file, load_list_from_file

from rxn_cluster_token_prompt.onmt_utils.metrics import get_multiplier
from rxn_cluster_token_prompt.onmt_utils.utils import RetroFiles

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def reorder_retro_predictions_class_token(
    ground_truth_file: Union[str, Path],
    predictions_file: Union[str, Path],
    confidences_file: Union[str, Path],
    fwd_predictions_file: Union[str, Path],
    classes_predictions_file: Union[str, Path],
    n_class_tokens: int,
) -> None:
    """
    Reorder the retro-preditions generated from a class-token model.
    For each sample x, N samples are created where N is the number of class token used.
    The retro predictions are originally ordered like e.g.:
        '[0] x' ->  top1 prediction('[0] x')
                ->  top2 prediction('[0] x')
                ...
        '[1] x' ->  top1 prediction('[1] x')
                ->  top2 prediction('[1] x')
                ...
        ...
        '[N] x' ->  top1 prediction('[N] x')
                ->  top2 prediction('[N] x')
                ...
    Starting from the log likelihood on each prediction we reorder them token-wise to remove the token dependency.
    So the new predictions for x will be:
        x   -> sorted([top1 prediction('[i] x') for i in number_class_tokens])
            -> sorted([top2 prediction('[i] x') for i in number_class_tokens])
            ...
    """
    logger.info(
        f'Reordering file "{predictions_file}", based on {n_class_tokens} class tokens.'
    )

    # We load the files and chunk the confidences
    ground_truth = load_list_from_file(ground_truth_file)
    predictions = load_list_from_file(predictions_file)
    confidences = load_list_from_file(confidences_file)
    fwd_predictions = load_list_from_file(fwd_predictions_file)
    classes_predictions = load_list_from_file(classes_predictions_file)

    # Get the exact multiplier
    multiplier = get_multiplier(ground_truth=ground_truth, predictions=predictions)

    if multiplier % n_class_tokens != 0:
        raise ValueError(
            f"The number of predictions ('{multiplier}') is not an exact "
            f"multiple of the number of class tokens '({n_class_tokens})'"
        )
    topx_per_class_token = int(multiplier / n_class_tokens)
    predictions_and_confidences = zip(
        predictions, confidences, fwd_predictions, classes_predictions
    )

    predictions_and_confidences_chunks = chunker(
        predictions_and_confidences, chunk_size=multiplier
    )

    # we will reorder the predictions class-token wise using the confidence
    predictions_and_confidences_reordered: List[Tuple[str, str, str, str]] = []

    for pred_and_conf in predictions_and_confidences_chunks:
        for topn in range(topx_per_class_token):
            # For each class token take the topn prediction and reorder them based on the
            # (negative) confidence (index x[1])
            topn_per_class_token = [
                chunk[topn]
                for chunk in chunker(pred_and_conf, chunk_size=topx_per_class_token)
            ]
            reordered = sorted(
                topn_per_class_token, key=lambda x: float(x[1]), reverse=True
            )
            predictions_and_confidences_reordered.extend(reordered)

    dump_list_to_file(
        (pred for pred, _, _, _ in predictions_and_confidences_reordered),
        str(predictions_file) + RetroFiles.REORDERED_FILE_EXTENSION,
    )
    dump_list_to_file(
        (conf for _, conf, _, _ in predictions_and_confidences_reordered),
        str(confidences_file) + RetroFiles.REORDERED_FILE_EXTENSION,
    )
    dump_list_to_file(
        (fwd_pred for _, _, fwd_pred, _ in predictions_and_confidences_reordered),
        str(fwd_predictions_file) + RetroFiles.REORDERED_FILE_EXTENSION,
    )
    dump_list_to_file(
        (
            classes_pred
            for _, _, _, classes_pred in predictions_and_confidences_reordered
        ),
        str(classes_predictions_file) + RetroFiles.REORDERED_FILE_EXTENSION,
    )


@click.command()
@click.option(
    "--ground_truth_file", "-g", required=True, help="File with ground truth."
)
@click.option(
    "--predictions_file", "-p", required=True, help="File with the predictions."
)
@click.option(
    "--confidences_file", "-l", required=True, help="File with the confidences."
)
@click.option(
    "--fwd_predictions_file",
    "-f",
    required=True,
    help="File with the forward predictions.",
)
@click.option(
    "--classes_predictions_file",
    "-c",
    required=True,
    help="File with the classes predictions.",
)
@click.option(
    "--n_class_tokens", "-n", required=True, type=int, help="Number of class tokens."
)
def main(
    ground_truth_file: str,
    predictions_file: str,
    confidences_file: str,
    fwd_predictions_file: str,
    classes_predictions_file: str,
    n_class_tokens: int,
) -> None:
    logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level="INFO")

    # Note: we put the actual code in a separate function, so that it can be
    # called also as a Python function.
    reorder_retro_predictions_class_token(
        ground_truth_file=ground_truth_file,
        predictions_file=predictions_file,
        confidences_file=confidences_file,
        fwd_predictions_file=fwd_predictions_file,
        classes_predictions_file=classes_predictions_file,
        n_class_tokens=n_class_tokens,
    )


if __name__ == "__main__":
    main()