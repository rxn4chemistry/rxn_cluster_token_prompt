# LICENSED INTERNAL CODE. PROPERTY OF IBM.
# IBM Research Zurich Licensed Internal Code
# (C) Copyright IBM Corp. 2021
# ALL RIGHTS RESERVED

from argparse import Namespace
from typing import Any, Iterable, Iterator, List, Optional, Union

from rxn_cluster_token_prompt.onmt_utils.internal_translation_utils import RawTranslator, TranslationResult, get_onmt_opt


class Translator:
    """
    Wraps the OpenNMT translation functionality into a class.
    """

    def __init__(self, opt: Namespace):
        """
        Should not be called directly as implementation may change; call the
        classmethods from_model_path or from_opt instead.
        Args:
            opt: model options.
        """
        self.onmt_translator = RawTranslator(opt=opt)

    def translate_single(self, sentence: str) -> str:
        """
        Translate one single sentence.
        """
        translations = self.translate_sentences([sentence])
        assert len(translations) == 1
        return translations[0]

    def translate_single_with_score(self, sentence: str) -> TranslationResult:
        """
        Translate one single sentence.
        """
        translations = self.translate_sentences_with_scores([sentence])
        assert len(translations) == 1
        return translations[0]

    def translate_sentences(self, sentences: Iterable[str]) -> List[str]:
        """
        Translate multiple sentences.
        """
        translations = self.translate_multiple_with_scores(sentences)
        return [t[0].text for t in translations]

    def translate_sentences_with_scores(self, sentences: Iterable[str]) -> List[TranslationResult]:
        """
        Translate multiple sentences.
        """
        translations = self.translate_multiple_with_scores(sentences)
        return [t[0] for t in translations]

    def translate_multiple_with_scores(
        self,
        sentences: Iterable[str],
        n_best: Optional[int] = None
    ) -> Iterator[List[TranslationResult]]:
        """
        Translate multiple sentences.
        Args:
            sentences: Sentences to translate.
            n_best: if provided, will overwrite the number of predictions to make.
        """
        additional_opt_kwargs = {}
        if n_best is not None:
            additional_opt_kwargs["n_best"] = n_best

        translations = self.onmt_translator.translate_sentences_with_onmt(
            sentences, **additional_opt_kwargs
        )

        return translations

    @classmethod
    def from_model_path(cls, model_path: Union[str, Iterable[str]], **kwargs: Any) -> "Translator":
        """
        Create a Translator instance from the model path(s).
        Args:
            model_path: path to the translation model file(s).
                If multiple are given, will be an ensemble model.
            kwargs: Additional values to be parsed for instantiating the translator,
                such as n_best, beam_size, max_length, etc.
        """
        if isinstance(model_path, str):
            model_path = [model_path]
        opt = get_onmt_opt(translation_model=list(model_path), **kwargs)
        return cls(opt=opt)

    @classmethod
    def from_opt(cls, opt: Namespace) -> "Translator":
        """
        Create a Translator instance from the opt arguments.
        Args:
            opt: model options.
        """
        return cls(opt=opt)
