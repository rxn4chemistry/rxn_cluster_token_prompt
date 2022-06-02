from enum import Enum
from typing import List

import click
import logging
import pandas as pd

from pathlib import Path
from rxn_utilities.file_utilities import dump_list_to_file
from rxn_chemutils.tokenization import tokenize_smiles

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logging.basicConfig(format="[%(asctime)s %(levelname)s] %(message)s", level=logging.INFO)


class ModelType(Enum):
    forward = "forward"
    retro = "retro"
    classification = "classification"


def tokenize_smiles_list(smiles_list: List[str]) -> List[str]:
    return [tokenize_smiles(smiles) for smiles in smiles_list]


@click.command()
@click.argument('input_csv_file', type=click.Path(exists=True), required=True)
@click.argument('output_path', type=click.Path(exists=True), required=True)
@click.option('--model-type', type=str, default="retro")
@click.option('--rxn-column-name', type=str, default="rxn")
@click.option('--class-column-name', type=str, default="class")
@click.option('--cluster-column-name', type=str, default="cluster")
@click.option('--seed', type=int, default=5)
@click.option('--split_ratio', default=0.1, type=float)
@click.option('--baseline', default=False, is_flag=True)
def main(input_csv_file: str, output_path: str, model_type: str, rxn_column_name: str, class_column_name: str,
         cluster_column_name: str, seed: int, split_ratio: float, baseline: bool):
    """Script to generate multiple random splits for a dataset

    Parameters
    ----------
    input_csv_file: a csv file containing the reaction smiles and the class information
    output_path: where to store the output files
    model_type: forward, retro or classification
    rxn_column_name: column under which reaction SMILES are stored
    class_column_name: column under which the reaction classes are stored
    cluster_column_name: column under which the clustering ids are stored
    seed: the seed used for generating the split
    split_ratio: the test and valid set ratio
    baseline: whether the model is the baseline model (in this case no class tokens are added to retro files)
    """
    df = pd.read_csv(input_csv_file)
    print(df.head())

    # Checks
    if rxn_column_name not in df.columns:
        raise KeyError(
            f"The column '{rxn_column_name}' was not found in the data."
        )

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
    logger.info("Generating the splits ...")
    test_df = df.sample(frac=split_ratio, random_state=seed)
    train_valid_df = df.drop(test_df.index)
    valid_df = train_valid_df.sample(frac=split_ratio, random_state=seed)
    train_df = train_valid_df.drop(valid_df.index)

    # Save the new dataframes to output_path/random${seed}
    new_output_path = Path(output_path) / f"random{seed}"
    new_output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Saving the dataframe splits ...")
    test_df.to_csv(new_output_path / "df.test.csv", index=False)
    train_valid_df.to_csv(new_output_path / "df.train-with-valid.csv", index=False)
    valid_df.to_csv(new_output_path / "df.valid.csv", index=False)
    train_df.to_csv(new_output_path / "df.train.csv", index=False)

    # Save the files to train and test forward model, classification model and retro chemistry puppeteer model
    logger.info("Generating the output folders for the models' files ...")

    # Model path
    model_output_path = new_output_path / model.name
    model_output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Generating the files ...")
    for df, split in zip([train_df, test_df, valid_df, train_valid_df], ['train', 'test', 'valid', 'train-with-valid']):
        if model == ModelType.classification:
            dump_list_to_file(tokenize_smiles_list(df[rxn_column_name]), model_output_path / f"rxn-{split}.txt")
            dump_list_to_file(df[class_column_name], model_output_path / f"class-{split}.txt")
        else:
            df['product'] = df[rxn_column_name].apply(lambda x: x.split('>>')[1])
            df['precursors'] = df[rxn_column_name].apply(lambda x: x.split('>>')[0])

            if model == ModelType.forward or baseline:
                dump_list_to_file(tokenize_smiles_list(df['product']), model_output_path / f"product-{split}.txt")
                dump_list_to_file(tokenize_smiles_list(df['precursors']), model_output_path / f"precursors-{split}.txt")
            else:
                # concatenate the class token for train and train-with-valid splits
                if split not in ['train', 'train-with-valid']:
                    df['retro_product'] = df['product']
                else:
                    df['retro_product'] = df.apply(lambda x: f"[{x[cluster_column_name]}]{x['product']}", axis=1)

                dump_list_to_file(tokenize_smiles_list(df['retro_product']), model_output_path / f"product-{split}.txt")
                dump_list_to_file(tokenize_smiles_list(df['precursors']), model_output_path / f"precursors-{split}.txt")

    logger.info(f"Output folder: {model_output_path}")


if __name__ == "__main__":
    main()
