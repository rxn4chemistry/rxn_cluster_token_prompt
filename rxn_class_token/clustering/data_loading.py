import logging
import os
import pickle
from pathlib import Path

import pandas as pd

from .fingerprints import generate_fps

# logger = logging.getLogger(__name__)
# logger.addHandler(logging.NullHandler())

FP_COLUMN = 'fps'
RXN_SMILES_COLUMN = 'rxn'

fps_model_path = '/Users/ato/Desktop/Git/rxnfp/rxnfp/models/transformers/bert_ft'  # The path to the trained fingerprints model
# data_csv_path = '/Users/ato/Desktop/df.dummy.with-reagents.valid.csv'
# fps_save_path = '/Users/ato/Desktop/df.dummy.with-reagents.valid.fps.pkl'
data_csv_path = '/Users/ato/Library/CloudStorage/Box-Box/IBM RXN for Chemistry/Data/class_token/std_pistachio_201002/data.csv'
fps_save_path = '/Users/ato/Library/CloudStorage/Box-Box/IBM RXN for Chemistry/Data/class_token/std_pistachio_201002/data.fps.pkl'


def ensure_fp(df: pd.DataFrame, saved_fp_path: Path) -> None:
    """Add the fingerprints to the DataFrame, compute them if did not exist."""
    if not saved_fp_path.exists():
        fps = generate_fps(model=fps_model_path, reaction_smiles=df[RXN_SMILES_COLUMN].tolist(), verbose=True)
        with open(saved_fp_path, 'wb') as f:
            pickle.dump(fps, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(saved_fp_path, 'rb') as f:
        fps_list = pickle.load(f)

    df[FP_COLUMN] = fps_list


def load_df() -> pd.DataFrame:
    df = pd.read_csv(data_csv_path)
    ensure_fp(df, Path(fps_save_path))  # Path(os.environ['FPS_SAVE_PATH'])
    return df
