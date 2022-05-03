import click
import pandas as pd
import json
import os
from enum import Enum
from rxn_chemutils.tokenization import tokenize_smiles


class Split(Enum):
    train = "train"
    train_with_valid = "train-with-valid"
    valid = "valid"
    test = "test"


@click.command()
@click.argument('input_csv_file', type=click.Path(exists=True), required=True)
@click.argument('output_path', type=click.Path(exists=True), required=True)
@click.option('--reaction_column_name', type=str, default='rxn')
@click.option('--classes_column_name', type=str, default='classes')
@click.option('--class-token/--no-class-token', required=False, default=True)
@click.option('--map-file', type=str, required=False)
@click.option(
    '--output-type',
    default='train',
    type=click.Choice(['train', 'train-with-valid', 'test', 'valid'], case_sensitive=True)
)
@click.option('--tokenize', type=bool, default=True)
def generate_dataset(
    input_csv_file: str, output_path: str, reaction_column_name: str, classes_column_name: str,
    class_token: bool, map_file: str, output_type: str, tokenize: bool
):
    """
    Script to generate the files needed for training and testing the class_token model.
    Also the files for the baseline model (no class tokens), can be generated.

    Parameters
    ----------
    input_csv_file: a csv file containing at least the reaction smiles
    output_path: where to store the output files
    reaction_column_name: the name of the column in the csv containing the reaction smiles
        defaults to 'rxn'
    classes_column_name: the name of the column in the csv containing the classes of each
        reaction, defaults to 'classes'
    class_token: wheater to generate the files for the class token model
    map_file: a json file containing the mapping from the 'classes' of the reactions
        to the wanted tokens
    output_type: train, test, valid, train-with-valid
    tokenize: whether to tokenize the output. Recomended.
    """

    df = pd.read_csv(input_csv_file)

    if reaction_column_name not in df.columns:
        raise KeyError(
            f"Sorry, we could not find a column in the dataframe named {reaction_column_name}!"
        )

    df['product'] = df[f"{reaction_column_name}"].apply(lambda x: x.split('>>')[1].strip())
    df['precursors'] = df[f"{reaction_column_name}"].apply(lambda x: x.split('>>')[0].strip())

    if tokenize:
        df['product'] = df['product'].apply(lambda x: tokenize_smiles(x))
        df['precursors'] = df['precursors'].apply(lambda x: tokenize_smiles(x))

    # Just save the (tokenized) product and precursors files: used for baseline model
    if not class_token:
        with open(os.path.join(output_path, f"precursors-{output_type}.txt"), 'w') as f:
            f.write('\n'.join(df['precursors'].values))
        with open(os.path.join(output_path, f"product-{output_type}.txt"), 'w') as f:
            f.write('\n'.join(df['product'].values))

        return 0

    if classes_column_name not in df.columns:
        raise KeyError(
            f"Sorry, we could not find a column in the dataframe named {classes_column_name}!"
        )

    # If json map file is given, map the NameRXN reaction class to the correct token
    # Otherwise look for a column providing directly the class tokens
    if map_file:
        with open(map_file) as f:
            classes_map = json.load(f)

        unique_values = sorted(set(classes_map.values()), key=lambda x: int(x))
        print("Tokens for class_token model:", unique_values)

        # Save the reactions with the token in front
        class_token_products = [
            f"[{classes_map[str(df[f'{classes_column_name}'].values[i])]}] {df['product'].values[i]}"
            for i in range(len(df))
        ]
        class_token_precursors = df.precursors.values

    else:
        if 'cluster_id' not in df.columns:
            raise KeyError("Sorry, we could not find a column in the dataframe named cluster_id!")

        unique_values = sorted(set(df['cluster_id'].values))
        print("Tokens for class_token model:", unique_values)

        # Save the reactions with the token in front
        class_token_products = [
            f"[{df['cluster_id'].values[i]}] {df['product'].values[i]}" for i in range(len(df))
        ]
        class_token_precursors = df.precursors.values

    with open(os.path.join(output_path, f"precursors-{output_type}.txt"), 'w') as f:
        f.write('\n'.join(class_token_precursors))
    with open(os.path.join(output_path, f"product-{output_type}.txt"), 'w') as f:
        f.write('\n'.join(class_token_products))

    # If the reactions are used for testing/validation, I need also a file where
    # for the same product I have multiple repetitions, as many as the number of
    # class tokens
    if output_type in [Split.valid.value, Split.test.value]:
        class_token_products = [
            f"[{i}] {elem}" for elem in df['product'].values for i in unique_values
        ]
        class_token_precursors = [elem for elem in df['precursors'].values for i in unique_values]

        with open(os.path.join(output_path, f"precursors-{output_type}-extended.txt"), 'w') as f:
            f.write('\n'.join(class_token_precursors))
        with open(os.path.join(output_path, f"product-{output_type}-extended.txt"), 'w') as f:
            f.write('\n'.join(class_token_products))


if __name__ == "__main__":
    generate_dataset()
