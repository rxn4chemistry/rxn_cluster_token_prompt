"""
Loads synthetic reaction datasets from USPTO.
This file contains loaders for synthetic reaction datasets from the US Patent Office. http://nextmovesoftware.com/blog/2014/02/27/unleashing-over-a-million-reactions-into-the-wild/.
"""
import logging

import pandas as pd
from rdkit import Chem

from rxn_cluster_token_prompt.repo_utils import data_directory
from rxn_cluster_token_prompt.utils import download_url

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logging.basicConfig(format="[%(asctime)s %(levelname)s] %(message)s", level=logging.INFO)

DEFAULT_DIR = data_directory() / 'uspto'

USPTO_URLS = {
    "USPTO_MIT": "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/USPTO_MIT.csv",
    "USPTO_STEREO": "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/USPTO_STEREO.csv",
    "USPTO_50K": "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/USPTO_50K.csv",
    "USPTO_FULL": "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/USPTO_FULL.csv",
}


def canonicalize_rxn(smiles: str) -> str:
    precursors, product = smiles.split(">>")
    try:
        return f"{Chem.MolToSmiles(Chem.MolFromSmiles(precursors))}>>{Chem.MolToSmiles(Chem.MolFromSmiles(product))}"
    except:
        logger.info(f"Cannot canonicalize reaction: {smiles}")
        return ""


class USPTOLoader:

    def __init__(self, dataset_name: str):
        if dataset_name not in list(USPTO_URLS.keys()):
            raise ValueError(f"The provided dataset name {dataset_name} is not available.")
        self.dataset_name = dataset_name
        self.dataset_url = USPTO_URLS[self.dataset_name]

    def download_dataset(self):
        if not (DEFAULT_DIR / f"{self.dataset_name}.csv").exists():
            logger.info("Downloading dataset...")
            download_url(url=self.dataset_url, dest_dir=DEFAULT_DIR, name=f"{self.dataset_name}.csv")
            logger.info("Dataset download complete.")

    def process_dataset(self, canonicalize: bool = True, single_product: bool = True):
        if not (DEFAULT_DIR / f"{self.dataset_name}.csv").exists():
            raise ValueError("The dataset was not found!")
        df = pd.read_csv(DEFAULT_DIR / f"{self.dataset_name}.csv")
        reactions_column_name = "reactions"

        if single_product:
            logger.info(f"Number of reactions: {len(df)}")
            logger.info("Removing reactions with more/less than 1 product ...")
            df["product"] = df[reactions_column_name].apply(lambda x: x.split('>>')[1])
            df_filter = df["product"].apply(lambda x: len(x.split('.')) == 1)
            df = df[df_filter]
            logger.info(f"Number of reactions: {len(df)}")
        if canonicalize:
            logger.info("Removing reactions which are not canonicalizable ...")
            df[f"{reactions_column_name}_can"] = df[reactions_column_name].apply(lambda x: canonicalize_rxn(x))
            df = df.loc[df[f"{reactions_column_name}_can"] != ""]
            logger.info(f"Number of reactions: {len(df)}")

        logger.info("Removing duplicate reactions ...")
        df = df.drop_duplicates().reset_index(drop=True)
        logger.info(f"Number of reactions: {len(df)}")
        df.to_csv(DEFAULT_DIR / f"{self.dataset_name}_processed.csv", index=False)
