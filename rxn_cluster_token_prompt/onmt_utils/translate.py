import logging
import subprocess
from typing import List, Optional

from rxn.utilities.files import PathLike, iterate_lines_from_file

from rxn_cluster_token_prompt.onmt_utils.translator import Translator

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def translate(
    model: PathLike,
    src: PathLike,
    tgt: Optional[PathLike],
    output: PathLike,
    n_best: int,
    beam_size: int,
    max_length: int,
    batch_size: int,
    gpu: bool,
    as_external_command: bool,
) -> None:
    """
    Run translate script.
    This is independent of any chemistry! As such, this does not take care of
    any tokenization either.
    This currently launches a subprocess relying on the OpenNMT binaries.
    In principle, the same could be achieved from Python code directly.
    Args:
        model: model checkpoint(s) to use.
        src: pointer to the file containing the source.
        tgt: pointer to the file containing the target, for calculation of the gold score.
        output: pointer to the file where to save the predictions.
        n_best: how many predictions to make per input.
        beam_size: beam size.
        max_length: max sequence length.
        batch_size: batch size for the prediction.
        gpu: whether to run the prediction on GPU.
        as_external_command: runs the onmt command instead of Python code.
    """
    if not gpu:
        logger.warning("GPU option not set. Only CPUs will be used. The translation may be slow!")

    if as_external_command:
        fn = translate_as_external_command
    else:
        fn = translate_as_python_code

    fn(
        model=model,
        src=src,
        tgt=tgt,
        output=output,
        n_best=n_best,
        beam_size=beam_size,
        max_length=max_length,
        batch_size=batch_size,
        gpu=gpu,
    )

    logger.info("Translation successful.")


def translate_as_python_code(
    model: PathLike,
    src: PathLike,
    tgt: Optional[PathLike],
    output: PathLike,
    n_best: int,
    beam_size: int,
    max_length: int,
    batch_size: int,
    gpu: bool,
) -> None:
    """
    Translate directly from Python - not by executing the OpenNMT command as a subprocess.
    See the function translate() for the documentation of the arguments.
    """
    logger.info(f'Running translation "{src}" -> "{output}", directly from Python code.')

    if tgt is not None:
        # Note: the gold score is determined by comparing with the provided tgt.
        # This is not supported at the moment, when running from the Python code.
        logger.warning(
            "No gold scores can be calculated at the moment "
            "when translating directly in Python."
        )

    translator = Translator.from_model_path(
        model_path=str(model),
        beam_size=beam_size,
        max_length=max_length,
        batch_size=batch_size,
        gpu=0 if gpu else -1,
    )

    src_iterator = iterate_lines_from_file(src)
    results_iterator = translator.translate_multiple_with_scores(src_iterator, n_best=n_best)

    # Note: this corresponds to the name of our OpenNMT fork
    log_probs_filename = str(output) + "_log_probs"

    with open(output, "wt") as f_tgt:
        with open(log_probs_filename, "wt") as f_lp:
            for result_list in results_iterator:
                for result in result_list:
                    f_tgt.write(f"{result.text}\n")
                    f_lp.write(f"{result.score}\n")


def translate_as_external_command(
    model: PathLike,
    src: PathLike,
    tgt: Optional[PathLike],
    output: PathLike,
    n_best: int,
    beam_size: int,
    max_length: int,
    batch_size: int,
    gpu: bool,
) -> None:
    """
    Translate by executing the OpenNMT command as a subprocess.
    See the function translate() for the documentation of the arguments.
    """

    if not gpu:
        logger.warning(
            "Running translation on CPU as a subprocess. Be careful "
            "when executing on a cluster: the subprocess may try to access "
            "all available cores."
        )

    command: List[str] = [
        "onmt_translate",
        "-model",
        str(model),
        "-src",
        str(src),
        "-output",
        str(output),
        "-log_probs",
        "-n_best",
        str(n_best),
        "-beam_size",
        str(beam_size),
        "-max_length",
        str(max_length),
        "-batch_size",
        str(batch_size),
    ]
    if tgt is not None:
        command.extend(["-tgt", str(tgt)])
    if gpu:
        command.extend(["-gpu", "0"])

    command_str = " ".join(command)
    logger.info(f"Running translation with command: {command_str}")
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        exception_str = f'The command "{command_str}" failed.'
        logger.error(exception_str)
        raise RuntimeError(exception_str) from e
