import logging
import os
import pickle
from pathlib import Path

import pandas as pd

from .fingerprints import generate_pistachio_fps, generate_1k_tpl_fps
from .standardize import standardize_for_fp_model

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

TPL_FP_COLUMN = 'tpl_fp'
PISTACHIO_FP_COLUMN = 'pistachio_fp'
RXN_SMILES_COLUMN = 'std_rxn_smiles'

xxx_dir = Path(os.environ['XXX_BOX_DIR'])
data_path = xxx_dir / 'Results' / 'diversity'

schneider_csv = data_path / 'schneider50k.csv'
std_schneider_csv = data_path / 'std_schneider50k.csv'
fp_tpl_50k_file = data_path / 'bert_class_1k_tpl_fps_schneider50k.pkl'
fp_pistachio_50k_file = data_path / 'rxnfp_pistachio_fps_schneider50k.pkl'

yyy_csv = data_path / 'yyy.csv'
std_yyy_csv = data_path / 'std_yyy.csv'
fp_tpl_yyy_file = data_path / 'bert_class_1k_tpl_fps_yyy.pkl'
fp_pistachio_yyy_file = data_path / 'rxnfp_pistachio_fps_yyy.pkl'

zzz_csv = data_path / 'zzz.csv'
std_zzz_csv = data_path / 'std_zzz.csv'
fp_tpl_zzz_file = data_path / 'bert_class_1k_tpl_fps_zzz.pkl'
fp_pistachio_zzz_file = data_path / 'rxnfp_pistachio_fps_zzz.pkl'


def ensure_standardized_dataframe(
    standardized_csv: Path, csv: Path, raw_rxn_column: str
) -> pd.DataFrame:
    """
    Load standardized DataFrame, save its csv if it didn't exist.

    Args:
        standardized_csv: CSV with standardized rxn SMILES.
        csv: original CSV.
        raw_rxn_column: reaction column in original CSV.
    """
    if not standardized_csv.exists():
        logger.info(f'{standardized_csv} does not existing yet. Standardizing now.')
        df = pd.read_csv(csv)
        df[RXN_SMILES_COLUMN] = df[raw_rxn_column].apply(standardize_for_fp_model)
        df.to_csv(standardized_csv, index=False)
        logger.info('Standardization done.')

    return pd.read_csv(standardized_csv)


def ensure_tpl_fp(df: pd.DataFrame, saved_tpl_fp_path: Path) -> None:
    """Add the TPL fingerprints to the DataFrame, compute them if did not exist."""
    if not saved_tpl_fp_path.exists():
        fps = generate_1k_tpl_fps(df[RXN_SMILES_COLUMN].tolist(), verbose=True)
        with open(saved_tpl_fp_path, 'wb') as f:
            pickle.dump(fps, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(saved_tpl_fp_path, 'rb') as f:
        fps_list = pickle.load(f)

    df[TPL_FP_COLUMN] = fps_list


def ensure_pistachio_fp(df: pd.DataFrame, saved_pistachio_fp_path: Path) -> None:
    """Add the Pistachio fingerprints to the DataFrame, compute them if did not exist."""
    if not saved_pistachio_fp_path.exists():
        fps = generate_pistachio_fps(df[RXN_SMILES_COLUMN].tolist(), verbose=True)
        with open(saved_pistachio_fp_path, 'wb') as f:
            pickle.dump(fps, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(saved_pistachio_fp_path, 'rb') as f:
        fps_list = pickle.load(f)

    df[PISTACHIO_FP_COLUMN] = fps_list


def load_schneider_df() -> pd.DataFrame:
    df = ensure_standardized_dataframe(
        std_schneider_csv, schneider_csv, 'rxnSmiles_Mapping_NameRxn'
    )
    ensure_tpl_fp(df, fp_tpl_50k_file)
    ensure_pistachio_fp(df, fp_pistachio_50k_file)
    return df


def load_yyy_df() -> pd.DataFrame:
    df = ensure_standardized_dataframe(std_yyy_csv, yyy_csv, 'rxn_smiles')
    ensure_tpl_fp(df, fp_tpl_yyy_file)
    ensure_pistachio_fp(df, fp_pistachio_yyy_file)
    return df


def load_zzz_df() -> pd.DataFrame:
    df = ensure_standardized_dataframe(std_zzz_csv, zzz_csv, 'rxn_smiles')
    ensure_tpl_fp(df, fp_tpl_zzz_file)
    ensure_pistachio_fp(df, fp_pistachio_zzz_file)
    return df
