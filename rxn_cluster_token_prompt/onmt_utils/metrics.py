from typing import Dict, List, Sequence, Tuple, TypeVar

import numpy as np
from rxn.utilities.containers import chunker

T = TypeVar("T")


def top_n_accuracy(ground_truth: Sequence[T], predictions: Sequence[T]) -> Dict[int, float]:
    """
    Compute the top-n accuracy values.
    Raises:
        ValueError: if the list sizes are incompatible, forwarded from get_multiplier().
    Returns:
        Dictionary of top-n accuracy values.
    """
    multiplier = get_multiplier(ground_truth=ground_truth, predictions=predictions)

    # we will count, for each "n", how many predictions are correct
    correct_for_topn: List[int] = [0 for _ in range(multiplier)]

    # We will process sample by sample - for that, we need to chunk the predictions
    prediction_chunks = chunker(predictions, chunk_size=multiplier)

    for gt, predictions in zip(ground_truth, prediction_chunks):
        for i in range(multiplier):
            correct = gt in predictions[:i + 1]
            correct_for_topn[i] += int(correct)

    return {i + 1: correct_for_topn[i] / len(ground_truth) for i in range(multiplier)}


def round_trip_accuracy(ground_truth: Sequence[T],
                        predictions: Sequence[T]) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Compute the round-trip accuracy values, split by n-th predictions.
    Raises:
        ValueError: if the list sizes are incompatible, forwarded from get_multiplier().
    Returns:
        Tuple of Dictionaries of round-trip accuracy "n" values and standard deviation (std_dev) "n" values.
        Here the standard deviation is the measure of how much the average round-trip accuracy can change from
        one sample to the other.
    """
    multiplier = get_multiplier(ground_truth=ground_truth, predictions=predictions)

    # we will get, for each prediction of each "n", how many predictions among the "n" are correct
    correct_for_n: List[List[int]] = [[] for _ in range(multiplier)]

    # We will process sample by sample - for that, we need to chunk the predictions
    prediction_chunks = chunker(predictions, chunk_size=multiplier)
    for gt, predictions in zip(ground_truth, prediction_chunks):
        correct_values = 0
        for i, prediction in enumerate(predictions):
            correct = gt == prediction
            correct_values += int(correct)
            correct_for_n[i].append(correct_values)

    # Note: for the "n"-th value, we must divide by "n=i+1" because the list elements were not averaged.
    accuracy = {i + 1: np.mean(correct_for_n[i]) / (i + 1) for i in range(multiplier)}
    std_dev = {i + 1: np.std(correct_for_n[i]) / (i + 1) for i in range(multiplier)}
    return accuracy, std_dev


def coverage(ground_truth: Sequence[T], predictions: Sequence[T]) -> Dict[int, float]:
    """
    Compute the coverage values, split by n-th predictions.
    Raises:
        ValueError: if the list sizes are incompatible, forwarded from get_multiplier().
    Returns:
        Dictionary of coverage "n" values.
    """
    multiplier = get_multiplier(ground_truth=ground_truth, predictions=predictions)

    # we will count, for each "n", if there is at list one correct prediction
    one_correct_for_n: List[int] = [0 for _ in range(multiplier)]

    # We will process sample by sample - for that, we need to chunk the predictions
    prediction_chunks = chunker(predictions, chunk_size=multiplier)

    for gt, predictions in zip(ground_truth, prediction_chunks):
        found_correct = 0
        for i, prediction in enumerate(predictions):
            if gt == prediction:
                found_correct = 1
            one_correct_for_n[i] += found_correct

    # Note: the total number of predictions to take into account for the "n"-th (= "i+1"th)
    # value is ALWAYS "len(ground_truth)".
    return {i + 1: one_correct_for_n[i] / len(ground_truth) for i in range(multiplier)}


def class_diversity(
    ground_truth: Sequence[T],
    predictions: Sequence[T],
    predicted_classes: Sequence[str],
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Compute the class diversity values, split by n-th predictions.
    Raises:
        ValueError: if the list sizes are incompatible, forwarded from get_multiplier().
    Returns:
        Tuple of Dictionaries of class diversity "n" values and standard deviation (std) "n" values.
        Here the standard deviation is the measure of how much the average class diversity can change from
        one sample to the other.
    """
    multiplier = get_multiplier(ground_truth=ground_truth, predictions=predictions)

    # we will count how many unique superclasses are present
    predicted_superclasses = [long_class.split(".")[0] for long_class in predicted_classes]

    # we will get, for each prediction of each "n", how many predictions among the "n" are correct
    classes_for_n: List[List[int]] = [[] for _ in range(multiplier)]

    # We will process sample by sample - for that, we need to chunk the predictions and the classes
    predictions_and_classes = zip(predictions, predicted_superclasses)
    prediction_and_classes_chunks = chunker(predictions_and_classes, chunk_size=multiplier)

    for gt, preds_and_classes in zip(ground_truth, prediction_and_classes_chunks):
        classes = set()
        for i, (pred, pred_class) in enumerate(preds_and_classes):
            if gt == pred and pred_class != "":
                classes.add(pred_class)
            classes_for_n[i].append(len(classes))

    # Note: the total number of predictions to take into account for the "n"-th (= "i+1"th)
    # value is "len(ground_truth)". A value < 1 is the consequence of having incorrect predictions
    classdiversity = {i + 1: np.mean(classes_for_n[i]) for i in range(multiplier)}
    std_dev = {i + 1: np.std(classes_for_n[i]) for i in range(multiplier)}
    return classdiversity, std_dev


def get_multiplier(ground_truth: Sequence[T], predictions: Sequence[T]) -> int:
    """
    Get the multiplier for the number of predictions by ground truth sample.
    Raises:
        ValueError: if the lists have inadequate sizes
    """
    n_gt = len(ground_truth)
    n_pred = len(predictions)

    if n_gt < 1 or n_pred < 1:
        raise ValueError(
            f"Inadequate number of predictions ({n_pred}) and/or ground truth samples ({n_gt})"
        )

    multiplier = n_pred // n_gt

    if n_pred != multiplier * n_gt:
        raise ValueError(
            f"The number of predictions ({n_pred}) is not an exact "
            f"multiple of the number of ground truth samples ({n_gt})"
        )

    return multiplier
