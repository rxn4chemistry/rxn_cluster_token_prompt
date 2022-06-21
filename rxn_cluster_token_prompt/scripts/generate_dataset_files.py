from enum import Enum
from typing import List

import click
import logging
import pandas as pd

from pathlib import Path
from rxn.utilities.files import dump_list_to_file
from rxn.chemutils.tokenization import tokenize_smiles
from rxn.utilities.logging import setup_console_logger

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ModelType(Enum):
    forward = "forward"
    retro = "retro"
    classification = "classification"


def tokenize_smiles_list(smiles_list: List[str]) -> List[str]:
    return [tokenize_smiles(smiles) for smiles in smiles_list]


@click.command()
@click.option(
    '--input_csv_file',
    '-i',
    type=click.Path(exists=True),
    required=True,
    help="Path to the csv file containing the reaction smiles and optionally the reaction class information."
)
@click.option(
    '--output_path', '-o', type=str, required=True, help="Output path"
)
@click.option(
    '--model-type',
    '-m',
    type=str,
    default="retro",
    help="Model type: available are 'retro'(default), 'forward' and 'classification'."
)
@click.option(
    '--rxn-column-name',
    type=str,
    default="rxn",
    help="Column under which reaction SMILES are stored, default is 'rxn'."
)
@click.option(
    '--class-column-name',
    type=str,
    default="class",
    help="Column under which the reaction classes are stored, default is 'class'."
)
@click.option(
    '--cluster-column-name',
    type=str,
    default="cluster_id",
    help="Column under which the clustering ids are stored, default is 'cluster_id'."
)
@click.option(
    '--seed', type=int, default=5, help="The seed used for generating the splits, default is 5."
)
@click.option(
    '--split_ratio', default=0.1, type=float, help="The test and valid set ratio, default is 0.1"
)
@click.option(
    '--baseline',
    default=False,
    is_flag=True,
    help="Whether the model files are for the baseline retro model (in this case no class tokens are added)"
)
def main(
    input_csv_file: str, output_path: str, model_type: str, rxn_column_name: str,
    class_column_name: str, cluster_column_name: str, seed: int, split_ratio: float, baseline: bool
):
    """Script to generate multiple random splits for a dataset for forward, retro or classification model
    training.
    """
    setup_console_logger()

    df = pd.read_csv(input_csv_file)
    print(df.head())

    # Checks
    if rxn_column_name not in df.columns:
        raise KeyError(f"The column '{rxn_column_name}' was not found in the data.")

    try:
        model = ModelType(model_type)
    except ValueError:
        raise ValueError(f"Attention, '{model_type}' is not a valid model type.")

    if model == ModelType.classification and class_column_name not in df.columns:
        raise KeyError(
            f"The column '{class_column_name}' was not found in the data. Cannot generate files for classification."
        )
    if not baseline and model == ModelType.retro and cluster_column_name not in df.columns:
        raise KeyError(
            f"The column '{cluster_column_name}' was not found in the data. Cannot generate files for retro."
        )

    # Generate test, valid and train random splits
    logger.info(f"Generating the splits with seed {seed} ...")
    test_df = df.sample(frac=split_ratio, random_state=seed)
    train_valid_df = df.drop(test_df.index)
    valid_df = train_valid_df.sample(frac=split_ratio, random_state=seed)
    train_df = train_valid_df.drop(valid_df.index)

    # Generates a new output path: output_path/random${seed}
    new_output_path = Path(output_path) / f"random{seed}"
    new_output_path.mkdir(parents=True, exist_ok=True)

    # Save the files to train and test forward model, classification model and retro chemistry puppeteer model
    logger.info("Generating the output folders for the models' files ...")

    # Model path
    model_output_path = new_output_path / model.name
    model_output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Generating the files ...")
    for df, split in zip(
        [train_df, test_df, valid_df, train_valid_df],
        ['train', 'test', 'valid', 'train-with-valid']
    ):
        if model == ModelType.classification:
            dump_list_to_file(
                tokenize_smiles_list(df[rxn_column_name]), model_output_path / f"rxn-{split}.txt"
            )
            dump_list_to_file(df[class_column_name], model_output_path / f"class-{split}.txt")
        else:
            df['product'] = df[rxn_column_name].apply(lambda x: x.split('>>')[1])
            df['precursors'] = df[rxn_column_name].apply(lambda x: x.split('>>')[0])

            if model == ModelType.forward or baseline:
                dump_list_to_file(
                    tokenize_smiles_list(df['product']), model_output_path / f"product-{split}.txt"
                )
                dump_list_to_file(
                    tokenize_smiles_list(df['precursors']),
                    model_output_path / f"precursors-{split}.txt"
                )
            else:
                # concatenate the class token for train and train-with-valid splits
                if split not in ['train', 'train-with-valid']:
                    df['retro_product'] = df['product']
                else:
                    df['retro_product'] = df.apply(
                        lambda x: f"[{x[cluster_column_name]}]{x['product']}", axis=1
                    )

                dump_list_to_file(
                    tokenize_smiles_list(df['retro_product']),
                    model_output_path / f"product-{split}.txt"
                )
                dump_list_to_file(
                    tokenize_smiles_list(df['precursors']),
                    model_output_path / f"precursors-{split}.txt"
                )

    logger.info(f"Output folder: {model_output_path}")


if __name__ == "__main__":
    main()
