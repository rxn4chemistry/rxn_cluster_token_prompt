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

# specify the variables
# FPS_SAVE_PATH The path where to store the computed fingerprints
# FPS_MODEL_PATH The path to the trained fingerprints model
# DATA_CSV_PATH The path to the data on which to compute the fingerprints

def ensure_fp(df: pd.DataFrame, saved_fp_path: Path) -> None:
    """Add the fingerprints to the DataFrame, compute them if did not exist."""
    if not saved_fp_path.exists():
        fps = generate_fps(model=os.environ['FPS_MODEL_PATH'], reaction_smiles=df[RXN_SMILES_COLUMN].tolist(), verbose=True)
        with open(saved_fp_path, 'wb') as f:
            pickle.dump(fps, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(saved_fp_path, 'rb') as f:
        fps_list = pickle.load(f)

    df[FP_COLUMN] = fps_list


def load_df() -> pd.DataFrame:
    df = pd.read_csv(Path(os.environ['DATA_CSV_PATH']))
    ensure_fp(df, Path(os.environ['FPS_SAVE_PATH']))
    return df
