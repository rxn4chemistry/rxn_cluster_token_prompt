"""Rxn cluster token prompt wrapper for the models."""
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from rxn.chemutils.conversion import canonicalize_smiles
from rxn.chemutils.tokenization import detokenize_smiles, tokenize_smiles

from rxn_cluster_token_prompt.onmt_utils.metrics import get_multiplier
from rxn_cluster_token_prompt.onmt_utils.translator import Translator
from rxn_cluster_token_prompt.onmt_utils.utils import (
    compute_probabilities,
    convert_class_token_idx_for_translation_models,
    create_rxn,
    maybe_canonicalize,
)
from rxn_cluster_token_prompt.repo_utils import models_directory

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

RETRO_MODEL_LOCATION_DICT = {
    "10clusters": models_directory() / "10clusters" / "10clusters.pt",
    "10clustersKmeans": models_directory() / "10clustersKmeans" / "10clustersKmeans.pt",
}

RETRO_MODEL_TOKENS_DICT = {"10clusters": 10, "10clustersKmeans": 10}

FORWARD_MODEL_LOCATION_DICT = {
    "forwardUSPTO": models_directory() / "forwardUSPTO" / "forwardUSPTO.pt"
}

CLASSIFICATION_MODEL_LOCATION_DICT = {
    "classificationUSPTO": models_directory()
    / "classificationUSPTO"
    / "classificationUSPTO.pt"
}


class RXNClusterTokenPrompt:
    """Wrap the Cluster Token Prompt Transformer model"""

    def __init__(
        self,
        retro_model_path: Union[str, Path] = RETRO_MODEL_LOCATION_DICT["10clusters"],
        forward_model_path: Union[str, Path] = FORWARD_MODEL_LOCATION_DICT[
            "forwardUSPTO"
        ],
        classification_model_path: Union[
            str, Path
        ] = CLASSIFICATION_MODEL_LOCATION_DICT["classificationUSPTO"],
        n_tokens: int = RETRO_MODEL_TOKENS_DICT["10clusters"],
        beam_size: int = 10,
        max_length: int = 300,
        batch_size: int = 64,
        n_best: int = 2,
    ):
        """
        RXNClusterTokenPrompt constructor.
        Args:
            retro_model_path: the path to the trained single-step retrosynthesis model
            forward_model_path: the path to the trained forward predictiion model
            classification_model_path: the path to the trained reaction classification model
            n_tokens: the number of cluster tokens in the chosen single-step retrosynthesis model
            beam_size: the beam for the models predictions
            max_length: maximum sequence length
            batch_size: the batch size for inference
            n_best: the number of predictions to retain for each used token prompt. Nees to be <= beam_size.
        Leave arguments empty to get default model
        """

        self.retro_model_path = retro_model_path
        self.forward_model_path = forward_model_path
        self.classification_model_path = classification_model_path
        self.n_tokens = n_tokens
        self.beam_size = beam_size
        self.max_length = max_length
        self.batch_size = batch_size
        self.n_best = n_best

        self.tokenizer = tokenize_smiles

    def retro_predict(
        self,
        products: List[str],
        canonicalize_input=True,
        canonicalize_output=True,
        remove_invalid_retro_predictions=True,
        probabilities=True,
        reorder_by_forward_likelihood=False,
        verbose=False,
    ) -> Dict[str, List[Tuple[Any, Any, str, Any, str]]]:
        """Function to run predictions with the RXNClusterTokenPrompt model.

        Args:
            products: products SMILES on which to run the predictions
            canonicalize_input: whether the input products are to be canonicalized
            canonicalize_output: whether the output of the model is to be canonicalized
            remove_invalid_retro_predictions: whether to remove the invalid retro predictions.
                if True, replaces the invalid predictions with an empty string
            probabilities: whether to compute probabilities from the log likelihood returned by the model
            reorder_by_forward_likelihood: whether to reorder the otput predictions by decreasing forward confidence
            verbose: allows for better display of the predictions

        Returns:
            A tuple so composed:
            (target, predicted_precursors, retro_confidence, predicted_product, forward_confidence, predicted_class)

        """
        if len(products) > 64:
            logger.info("Requesting translation on many products ...might be slow")
        if canonicalize_input:
            products = [canonicalize_smiles(product) for product in products]

        # Prompt preparation
        class_token_products = (
            f"{convert_class_token_idx_for_translation_models(class_token_idx)}{molecule}"
            for molecule in products
            for class_token_idx in range(self.n_tokens)
        )

        # Tokenization
        tok_products = [self.tokenizer(product) for product in class_token_products]

        retro_translator = Translator.from_model_path(
            model_path=str(self.retro_model_path),
            beam_size=self.beam_size,
            max_length=self.max_length,
            batch_size=self.batch_size,
            gpu=-1,
        )

        forward_translator = Translator.from_model_path(
            model_path=str(self.forward_model_path),
            beam_size=10,
            max_length=300,
            batch_size=self.batch_size,
            gpu=-1,
        )

        classification_translator = Translator.from_model_path(
            model_path=str(self.classification_model_path),
            beam_size=5,
            max_length=300,
            batch_size=self.batch_size,
            gpu=-1,
        )

        results_iterator = retro_translator.translate_multiple_with_scores(
            tok_products, n_best=self.n_best
        )

        # Collect results from retrosynthesis, forward and classification models
        res = []
        for result_list in results_iterator:
            for result in result_list:
                prediction = (
                    maybe_canonicalize(detokenize_smiles(result.text))
                    if canonicalize_output
                    else detokenize_smiles(result.text)
                )
                confidence = (
                    compute_probabilities(result.score)
                    if probabilities
                    else result.score
                )
                res.append((prediction, confidence))

        multiplier = get_multiplier(ground_truth=products, predictions=res)

        output: Dict[str, List[Tuple[Any, Any, str, Any, str]]] = {}

        for i, product in enumerate(products):

            def _remove_invalids(
                predictions: List[Tuple[str, Any]]
            ) -> List[Tuple[str, Any]]:
                return [pred for pred in predictions if pred[0] != ""]

            def _forward_and_classification_prediction(
                predictions: List[Tuple[str, Any]],
            ) -> List[Tuple[Any, Any, str, Any, str]]:
                enriched_predictions = []
                for prediction, confidence in predictions:
                    forward_output = forward_translator.translate_single_with_score(
                        tokenize_smiles(prediction)
                    )
                    round_trip_prediction, round_trip_confidence = (
                        maybe_canonicalize(detokenize_smiles(forward_output.text)),
                        compute_probabilities(forward_output.score)
                        if probabilities
                        else forward_output.score,
                    )
                    predicted_rxn = create_rxn(prediction, product)
                    classification_output = (
                        classification_translator.translate_single_with_score(
                            tokenize_smiles(predicted_rxn)
                        )
                    )
                    classification_prediction = classification_output.text
                    enriched_predictions.append(
                        (
                            prediction,
                            confidence,
                            round_trip_prediction,
                            round_trip_confidence,
                            classification_prediction,
                        )
                    )
                return enriched_predictions

            output[product] = (
                _forward_and_classification_prediction(
                    _remove_invalids(res[i : i + multiplier])
                )
                if remove_invalid_retro_predictions
                else _forward_and_classification_prediction(res[i : i + multiplier])
            )

            if reorder_by_forward_likelihood:
                output[product] = sorted(
                    output[product], key=lambda x: x[3], reverse=True
                )
            if verbose:
                print("Target molecule: ", product)
                for p in output[product]:
                    self._display(p)

        return output

    def _display(self, prediction: Tuple[Any, Any, str, Any, str]):
        """Diplays a prediction composed by
        (target, predicted_precursors, retro_confidence, predicted_product, forward_confidence, predicted_class)
        """
        print(
            {
                "predicted precursors": prediction[0],
                "retro confidence": prediction[1],
                "predicted product": prediction[2],
                "forward confidence": prediction[3],
                "predicted class": prediction[4],
            }
        )
