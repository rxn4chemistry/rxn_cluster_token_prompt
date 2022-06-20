import logging
import os
import shutil

import numpy as np

from rxn.chemutils.conversion import canonicalize_smiles
from rxn.chemutils.exceptions import InvalidSmiles

from rxn.chemutils.tokenization import (
    TokenizationError,
    detokenize_smiles,
    tokenize_smiles,
)
from rxn.utilities.files import (
    PathLike,
    dump_list_to_file,
    iterate_lines_from_file,
)


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def raise_if_identical_path(input_path: PathLike, output_path: PathLike) -> None:
    """
    Raise an exception if input and output paths point to the same file.
    """
    if os.path.realpath(input_path) == os.path.realpath(output_path):
        raise ValueError(
            f'The output path, "{output_path}", must be '
            f'different from the input path, "{input_path}".'
        )


def string_is_tokenized(smiles_line: str) -> bool:
    """
    Whether a line is tokenized or not.
    Args:
        smiles_line: line to inspect
    Raises:
        TokenizationError: propagated directly from tokenize_smiles()
    """
    detokenized = detokenize_smiles(smiles_line)
    tokenized = tokenize_smiles(detokenized)
    return smiles_line == tokenized


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


def convert_class_token_idx_for_translation_models(class_token_idx: int) -> str:
    return f"[{class_token_idx}]"


def tokenize_line(smiles_line: str, invalid_placeholder: str) -> str:
    try:
        return tokenize_smiles(smiles_line)
    except TokenizationError:
        logger.debug(f'Error when tokenizing "{smiles_line}"')
        return invalid_placeholder


def tokenize_file(
    input_file: PathLike, output_file: PathLike, invalid_placeholder: str = ""
) -> None:
    raise_if_identical_path(input_file, output_file)
    logger.info(f'Tokenizing "{input_file}" -> "{output_file}".')

    tokenized = (
        tokenize_line(line, invalid_placeholder)
        for line in iterate_lines_from_file(input_file)
    )

    dump_list_to_file(tokenized, output_file)


def detokenize_file(
    input_file: PathLike,
    output_file: PathLike,
) -> None:
    raise_if_identical_path(input_file, output_file)
    logger.info(f'Detokenizing "{input_file}" -> "{output_file}".')

    detokenized = (
        detokenize_smiles(line) for line in iterate_lines_from_file(input_file)
    )
    dump_list_to_file(detokenized, output_file)


def detokenize_class(tokenized_class: str) -> str:
    """
    Function performing a detokenization of the reaction class used in the Transformer classification
    model. E.g. '1 1.2 1.2.3' -> '1.2.3' or '1' -> '1' (unchanged) for USPTO
    Args:
        tokenized_class: str to detokenize
    Raises:
        ValueError: if the input string format is not correct
    """
    if tokenized_class == "0":
        return tokenized_class

    splitted_class = tokenized_class.split(" ")
    if len(splitted_class) == 1:
        if len(splitted_class[0].split(".")) == 3:
            # here the class is already detokenized
            return tokenized_class
        try:
            int(splitted_class[0])
            return tokenized_class
        except:
            pass

    if len(splitted_class) != 3:
        raise ValueError(
            f'The class to be detokenized, "{tokenized_class}", is probably not in the correct format.'
        )
    return splitted_class[-1]


def tokenize_class(detokenized_class: str) -> str:
    """
    Function performing a tokenization of the reaction class used in the Transformer classification
    model. E.g. '1.2.3' -> '1 1.2 1.2.3'
    Args:
        detokenized_class: str to tokenize
    Raises:
        ValueError: if the input string format is not correct
    """
    if detokenized_class == "0":
        return detokenized_class
    try:
        int(detokenized_class)
        return detokenized_class
    except:
        pass

    splitted_class = detokenized_class.split(".")
    if len(splitted_class) == 4 and len(detokenized_class.split(" ")) == 3:
        # here the class is already tokenized
        return detokenized_class
    if len(splitted_class) != 3:
        raise ValueError(
            f'The class to be tokenized, "{detokenized_class}", is probably not in the correct format.'
        )
    a, b, _ = splitted_class
    return f"{a} {a}.{b} {detokenized_class}"


def tokenize_class_line(class_line: str, invalid_placeholder: str) -> str:
    try:
        return tokenize_class(class_line)
    except ValueError:
        logger.debug(f'Error when tokenizing the class "{class_line}"')
        return invalid_placeholder


def detokenize_class_line(class_line: str, invalid_placeholder: str) -> str:
    try:
        return detokenize_class(class_line)
    except ValueError:
        logger.debug(f'Error when detokenizing the class "{class_line}"')
        return invalid_placeholder


def detokenize_classification_file(
    input_file: PathLike, output_file: PathLike, invalid_placeholder: str = ""
) -> None:
    raise_if_identical_path(input_file, output_file)
    logger.info(f'Detokenizing "{input_file}" -> "{output_file}".')

    detokenized = (
        detokenize_class_line(line, invalid_placeholder)
        for line in iterate_lines_from_file(input_file)
    )
    dump_list_to_file(detokenized, output_file)


def tokenize_classification_file(
    input_file: PathLike, output_file: PathLike, invalid_placeholder: str = ""
) -> None:
    raise_if_identical_path(input_file, output_file)
    logger.info(f'Tokenizing "{input_file}" -> "{output_file}".')

    tokenized = (
        tokenize_class_line(line, invalid_placeholder)
        for line in iterate_lines_from_file(input_file)
    )
    dump_list_to_file(tokenized, output_file)


def classification_string_is_tokenized(classification_line: str) -> bool:
    """
    Whether a classification line is tokenized or not.
    Args:
        classification_line: line to inspect
    Raises:
        ValueError: for errors in tokenization or detokenization
    """
    detokenized = detokenize_class(classification_line)
    tokenized = tokenize_class(detokenized)
    return classification_line == tokenized


def classification_file_is_tokenized(filepath: PathLike) -> bool:
    """
    Whether a file contains tokenized classes or not.
    '1.2.3' -> '1 1.2 1.2.3'
    By default, this looks at the first non-empty line of the file only!
    Raises:
        ValueError: for errors in tokenization or detokenization
        RuntimeError: for empty files or files with empty lines only.
    Args:
        filepath: path to the file.
    """
    for line in iterate_lines_from_file(filepath):
        # Ignore empty lines
        if line == "":
            continue
        return classification_string_is_tokenized(line)
    raise RuntimeError(
        f'Could not determine whether "{filepath}" is class-tokenized: empty lines only.'
    )


def file_is_tokenized(filepath: PathLike) -> bool:
    """
    Whether a file contains tokenized SMILES or not.
    By default, this looks at the first non-empty line of the file only!
    Raises:
        TokenizationError: propagated from tokenize_smiles()
        RuntimeError: for empty files or files with empty lines only.
    Args:
        filepath: path to the file.
    """
    for line in iterate_lines_from_file(filepath):
        # Ignore empty lines
        if line == "":
            continue
        return string_is_tokenized(line)
    raise RuntimeError(
        f'Could not determine whether "{filepath}" is tokenized: empty lines only.'
    )


def copy_as_detokenized(src: PathLike, dest: PathLike) -> None:
    """
    Copy a source file to a destination, while making sure that it is not tokenized.
    """
    if file_is_tokenized(src):
        logger.info(f'Copying and detokenizing "{src}" -> "{dest}".')
        detokenize_file(src, dest)
    else:
        logger.info(f'Copying "{src}" -> "{dest}".')
        shutil.copy(src, dest)


class MetricsFiles:
    def __init__(self, directory: PathLike):
        self.directory = Path(directory)
        self.log_file = self.directory / "log.txt"
        self.metrics_file = self.directory / "metrics.json"


class RetroFiles(MetricsFiles):
    """
    Class holding the locations of the files to write to or to read from for
    the evaluation of retro metrics.
    """

    REORDERED_FILE_EXTENSION = ".reordered"

    def __init__(self, directory: PathLike):
        super().__init__(directory=directory)
        self.gt_products = self.directory / "gt_products.txt"
        self.gt_precursors = self.directory / "gt_precursors.txt"
        self.class_token_products = self.directory / "class_token_products.txt"
        self.class_token_precursors = self.directory / "class_token_precursors.txt"
        self.predicted_precursors = self.directory / "predicted_precursors.txt"
        self.predicted_precursors_canonical = (
            self.directory / "predicted_precursors_canonical.txt"
        )
        self.predicted_precursors_log_probs = (
            self.directory / "predicted_precursors.txt.tokenized_log_probs"
        )
        self.predicted_products = self.directory / "predicted_products.txt"
        self.predicted_products_canonical = (
            self.directory / "predicted_products_canonical.txt"
        )
        self.predicted_products_log_probs = (
            self.directory / "predicted_products.txt.tokenized_log_probs"
        )
        self.predicted_rxn_canonical = self.directory / "predicted_rxn_canonical.txt"
        self.predicted_classes = self.directory / "predicted_classes.txt"
