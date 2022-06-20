import logging
import os
import pickle
from pathlib import Path

import pandas as pd

from .fingerprints import generate_fps

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.NullHandler())

FP_COLUMN = 'fps'


def ensure_fp(df: pd.DataFrame, saved_fp_path: Path) -> None:
    """Add the fingerprints to the DataFrame, compute them if did not exist."""
    if not saved_fp_path.exists():
        logger.info("Fingerprints not available. Computing them ...")
        fps = generate_fps(
            model=os.environ['FPS_MODEL_PATH'],
            reaction_smiles=df[os.environ['RXN_SMILES_COLUMN']].tolist(),
            verbose=True
        )
        with open(saved_fp_path, 'wb') as f:
            pickle.dump(fps, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Fingerprints computed. Saved in {saved_fp_path} .")

    with open(saved_fp_path, 'rb') as f:
        fps_list = pickle.load(f)
        logger.info(f"Fingerprints saved in {saved_fp_path} .")

    df[FP_COLUMN] = fps_list
    logger.info("Fingerprints loaded to dataframe.")


def load_df() -> pd.DataFrame:
    df = pd.read_csv(Path(os.environ['DATA_CSV_PATH']))
    logger.info("Ensuring fingerprints.")
    ensure_fp(df, Path(os.environ['FPS_SAVE_PATH']))
    return df
