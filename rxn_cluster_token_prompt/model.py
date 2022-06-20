"""Rxn cluster token prompt wrapper for the models."""
import logging

from rxn_cluster_token_prompt.onmt_utils.metrics import get_multiplier
from rxn_cluster_token_prompt.onmt_utils.translator import Translator
from rxn_cluster_token_prompt.repo_utils import models_directory
from typing import List, Dict, Tuple
from rxn.utilities.logging import setup_console_logger
from rxn.chemutils.tokenization import tokenize_smiles, detokenize_smiles
from rxn.chemutils.conversion import canonicalize_smiles
from rxn_cluster_token_prompt.onmt_utils.utils import maybe_canonicalize, compute_probabilities, create_rxn

from rxn_cluster_token_prompt.onmt_utils.utils import convert_class_token_idx_for_translation_models

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
setup_console_logger()

RETRO_MODEL_LOCATION_DICT = {
    "10clusters": models_directory() / "10clusters" / "10clusters.pt",
    "10clustersKmeans": models_directory() / "10clustersKmeans" / "10clustersKmeans.pt",
}

RETRO_MODEL_TOKENS_DICT = {
    "10clusters": 10,
    "10clustersKmeans": 10
}

FORWARD_MODEL_LOCATION_DICT = {
    "forwardUSPTO": models_directory() / "forwardUSPTO" / "forwardUSPTO.pt"
}

CLASSIFICATION_MODEL_LOCATION_DICT = {
    "classificationUSPTO": models_directory() / "classificationUSPTO" / "classificationUSPTO.pt"
}


class RXNClusterTokenPrompt:
    """ Wrap the Cluster Token Prompt Transformer model
    """

    def __init__(
            self,
            config: Dict = {},
    ):
        """
        RXNMapper constructor.
        Args:
            config (Dict): Config dict, leave it empty to have the
                default USPTO model.
        """

        # Config takes "retro_model_path", "forward_model_path", "classification_model_path" and other params
        self.config = config
        self.retro_model_path = self.config.get(
            "retro_model_path",
            RETRO_MODEL_LOCATION_DICT["10clusters"],
        )
        self.forward_model_path = self.config.get(
            "forward_model_path",
            FORWARD_MODEL_LOCATION_DICT["forwardUSPTO"],
        )
        self.classification_model_path = self.config.get(
            "classification_model_path",
            CLASSIFICATION_MODEL_LOCATION_DICT["classificationUSPTO"],
        )
        self.n_tokens = self.config.get("n_tokens", RETRO_MODEL_TOKENS_DICT["10clusters"])

        self.tokenizer = tokenize_smiles

    def retro_predict(self, products: List[str], canonicalize_input=True, canonicalize_output=True,
                      remove_invalid_retro_predictions=True, probabilities=True, reorder_by_forward_likelihood=False,
                      display=False) -> Dict[str, List[tuple]]:
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
            beam_size=self.config.get("beam_size", 10),
            max_length=self.config.get("max_length", 300),
            batch_size=self.config.get("batch_size", 64),
            gpu=-1
        )

        forward_translator = Translator.from_model_path(
            model_path=str(self.forward_model_path),
            beam_size=10,
            max_length=300,
            batch_size=self.config.get("batch_size", 64),
            gpu=-1
        )

        classification_translator = Translator.from_model_path(
            model_path=str(self.classification_model_path),
            beam_size=5,
            max_length=300,
            batch_size=self.config.get("batch_size", 64),
            gpu=-1
        )

        results_iterator = retro_translator.translate_multiple_with_scores(
            tok_products, n_best=self.config.get("n_best", 10)
        )

        # Collect results from retrosynthesis, forward and classification models
        res = []
        for result_list in results_iterator:
            for result in result_list:
                prediction = maybe_canonicalize(detokenize_smiles(result.text)) if canonicalize_output \
                    else detokenize_smiles(result.text)
                confidence = compute_probabilities(result.score) if probabilities else result.score
                res.append((prediction, confidence))

        multiplier = get_multiplier(ground_truth=products, predictions=res)

        output = {}

        for i, product in enumerate(products):

            def _remove_invalids(predictions: List[Tuple]) -> List[Tuple]:
                return [pred for pred in predictions if pred[0] != '']

            def _forward_and_classification_prediction(predictions: List[Tuple]) -> List[Tuple]:
                enriched_predictions = []
                for prediction, confidence in predictions:
                    forward_output = forward_translator.translate_single_with_score(
                        tokenize_smiles(prediction))
                    round_trip_prediction, round_trip_confidence = \
                        maybe_canonicalize(detokenize_smiles(forward_output.text)), \
                        compute_probabilities(forward_output.score) if probabilities else forward_output.score
                    predicted_rxn = create_rxn(prediction, product)
                    classification_output = classification_translator.translate_single_with_score(
                        tokenize_smiles(predicted_rxn))
                    classification_prediction = classification_output.text
                    enriched_predictions.append((prediction, confidence, round_trip_prediction, round_trip_confidence,
                                                 classification_prediction))
                return enriched_predictions

            output[product] = _forward_and_classification_prediction(_remove_invalids(res[i: i + multiplier])) \
                if remove_invalid_retro_predictions else _forward_and_classification_prediction(res[i: i + multiplier])
            if reorder_by_forward_likelihood:
                output[product] = sorted(output[product], key=lambda x: x[3], reverse=True)
            if display:
                print("Target molecule: ", product)
                for prediction in output[product]:
                    self._display(prediction)

        return output

    def _display(self, prediction: Tuple):
        """Diplays a prediction composed by
        (target, predicted_precursors, retro_confidence, predicted_product, forward_confidence, predicted_class)
        """
        print(
            {"predicted precursors": prediction[0],
             "retro confidence": prediction[1],
             "predicted product": prediction[2],
             "forward confidence": prediction[3],
             "predicted class": prediction[4]
             }
        )
