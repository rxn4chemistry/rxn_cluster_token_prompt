import copy
import os
import tempfile
from argparse import Namespace
from itertools import repeat
from typing import Any, Iterable, Iterator, List, Optional

import attr
import onmt.opts as opts
from onmt.translate.translator import build_translator
from onmt.utils.misc import split_corpus
from onmt.utils.parse import ArgumentParser


@attr.s(auto_attribs=True)
class TranslationResult:
    """
    Struct containing the result of a translation with OpenNMT.
    """

    text: str
    score: float


class RawTranslator:
    """
    Translator class that is very coupled to the internal OpenNMT implementation.
    """

    def __init__(self, opt: Namespace):
        self.opt = opt
        self.score_for_empty_input = -9999.9999
        self.dummy_string_for_empty_input = "C . C . C . C"

        # to avoid the creation of an unnecessary file
        out_file = open(os.devnull, "w")
        self.internal_translator = build_translator(
            self.opt, report_score=False, out_file=out_file
        )

    def translate_sentences_with_onmt(
        self, sentences: Iterable[str], **opt_updated_kwargs: Any
    ) -> Iterator[List[TranslationResult]]:
        """
        Do the translation (in tokenized format) with OpenNMT.
        Args:
            sentences: sentences to translate
            opt_updated_kwargs: values to update in the "opt" of the translator. The
                translator is not instantiated again from those values, therefore this
                only affects values that are used for translation, such as n_best.
        """
        new_opt = copy.deepcopy(self.opt)
        for key, value in opt_updated_kwargs.items():
            setattr(new_opt, key, value)
        with tempfile.NamedTemporaryFile() as tmp_src, tempfile.NamedTemporaryFile() as tmp_output:
            new_opt.src = tmp_src.name
            new_opt.output = tmp_output.name

            # List to track which inputs were empty, for post-processing
            empty_input: List[bool] = []

            # write source sentences to temporary input file
            with open(new_opt.src, "wt") as f:
                for sentence in sentences:
                    # In order to avoid problems with batches full of empty string on GPUs,
                    # we write a dummy line instead of the empty string. These lines
                    # are post-processed again below to replace the predictions by
                    # empty strings.
                    is_empty = False
                    if sentence == "":
                        sentence = self.dummy_string_for_empty_input
                        is_empty = True

                    f.write(f"{sentence}\n")
                    empty_input.append(is_empty)

            for translation_results, is_empty in zip(
                self.translate_with_onmt(new_opt), empty_input
            ):
                # For predictions corresponding to empty predictions, return
                # an empty string with adequate score
                if is_empty:
                    yield [
                        TranslationResult("", self.score_for_empty_input)
                        for _ in translation_results
                    ]
                else:
                    yield translation_results

    def translate_with_onmt(self, opt) -> Iterator[List[TranslationResult]]:
        """
        Do the translation (in tokenized format) with OpenNMT.
        Args:
            opt: args given to the main script
        Returns:
            Generator of TranslationResults; they will be yielded in chunks of
            size opt.shard_size.
        """
        # for some versions, it seems that n_best is not updated, we therefore do it manually here
        self.internal_translator.n_best = opt.n_best

        src_shards = split_corpus(opt.src, opt.shard_size)
        tgt_shards = (
            split_corpus(opt.tgt, opt.shard_size)
            if opt.tgt is not None
            else repeat(None)
        )
        shard_pairs = zip(src_shards, tgt_shards)

        for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
            l1, l2 = self.internal_translator.translate(
                src=src_shard,
                tgt=tgt_shard,
                src_dir=opt.src_dir,
                batch_size=opt.batch_size,
                batch_type=opt.batch_type,
                attn_debug=opt.attn_debug,
            )
            for score_list, translation_list in zip(l1, l2):
                yield [
                    TranslationResult(text=t, score=s.item())
                    for s, t in zip(score_list, translation_list)
                ]


def get_onmt_opt(
    translation_model: Iterable[str],
    src_file: Optional[str] = None,
    output_file: Optional[str] = None,
    **kwargs: Any,
) -> Namespace:
    """
    Create the opt arguments by taking the defaults and overwriting a few values.
    Args:
        translation_model: Model(s) to for translation
        src_file: Source file
        output_file: Output file
        kwargs: additional values to change in the resulting opt
    """

    # Some values are needed and must be parsed from args, other values can
    # simply be overwritten from the default ones
    src = src_file if src_file is not None else "(unused)"
    output = output_file if output_file is not None else "(unused)"
    args_str = f'--model {" ".join(translation_model)} --src {src} --output {output}'
    args = args_str.split()

    parser = onmt_parser()
    opt = parser.parse_args(args)
    for key, value in kwargs.items():
        setattr(opt, key, value)
    ArgumentParser.validate_translate_opts(opt)

    return opt


def onmt_parser() -> ArgumentParser:
    """
    Create the OpenNMT parser, adapted from OpenNMT-Py repo.
    """

    parser = ArgumentParser(description="translate.py")

    opts.config_opts(parser)
    opts.translate_opts(parser)

    return parser
