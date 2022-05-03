from rxn_class_token.repo_utils import data_directory
from rxn_class_token.scripts.generate_dataset import generate_dataset
from click.testing import CliRunner

# def test_train_baseline():
#     runner = CliRunner()
#     result = runner.invoke(
#         main,
#         input=f'{data_directory()}/test_data/df_sample.csv {data_directory()}/test_data/output'
#     )
#     print(result)
#     # assert result == ''
#
#


def test_wrong_reaction_column():
    runner = CliRunner()
    result = runner.invoke(
        generate_dataset, [
            f'{data_directory()}/test_data/df_sample.csv', f'{data_directory()}/test_data/output',
            '--reaction_column_name', 'reactions'
        ]
    )
    assert isinstance(result.exception, KeyError)


def test_wrong_classes_colun_name():
    runner = CliRunner()
    result = runner.invoke(
        generate_dataset, [
            f'{data_directory()}/test_data/df_sample.csv', f'{data_directory()}/test_data/output',
            '--reaction_column_name', 'rxn', '--classes_column_name', 'dummy_class'
        ]
    )
    assert isinstance(result.exception, KeyError)
