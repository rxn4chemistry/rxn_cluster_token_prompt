import logging
from typing import Union, List

import attr
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

from rxn_cluster_token_prompt.clustering.fingerprints import generate_fps
from rxn_cluster_token_prompt.clustering.standardize import standardize_for_fp_model

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@attr.s(auto_attribs=True)
class Clusterer:
    """Clusterer for fingerprints.

    Instances can be saved and loaded to/from disk relying on pickle.
    See https://scikit-learn.org/stable/modules/model_persistence.html.
    """
    pca: PCA
    kmeans: KMeans

    def _transform(self, data: np.ndarray) -> np.ndarray:
        return self.pca.transform(data)

    def predict(self, data: np.ndarray) -> List[int]:
        """Predict the cluster based on the fingerprints."""
        transformed = self._transform(data)
        return self.kmeans.predict(transformed)

    def get_cluster_nos(
        self,
        rxn_smiles_list: List[str],
        model_path: str,
        standardize: bool = True,
        verbose: bool = False
    ) -> List[int]:
        """Get the cluster numbers for a list of reaction SMILES.

        Will get the fingerprints on-the-go; hence, this function may be slow.

        NB: It is not efficient to call this function multiple times separately,
        as it will load the rxnfp model multiple times.

        Args:
            rxn_smiles_list: list of reaction SMILES.
            standardize: whether standardization of the SMILES stings is needed.
                Note: Likely to be needed anyway to replace the '~' by '.'!
            model_path: The path to the fps bert model
            verbose: whether to print the progress with tqdm.
        """
        n_reactions = len(rxn_smiles_list)

        # "converting" list to a generator
        rxn_smiles_iterator = (smiles for smiles in rxn_smiles_list)

        if standardize:
            rxn_smiles_iterator = (
                standardize_for_fp_model(smiles) for smiles in rxn_smiles_iterator
            )

        if verbose:
            rxn_smiles_iterator = (
                smiles for smiles in tqdm(
                    rxn_smiles_iterator,
                    desc='Standardizing and getting fingerprints',
                    total=n_reactions
                )
            )

        fps_list = generate_fps(
            model=model_path, reaction_smiles=rxn_smiles_iterator, verbose=False
        )

        logger.info('Obtained the fingerprints; getting clusters now.')
        fps = np.array(fps_list)
        return self.predict(fps)


class ClustererFitter:

    def __init__(
        self,
        data: np.ndarray,
        scaler,
        clusterer,
        fit_scaler_on: int = 10000,
        fit_clusterer_on: int = 10000,
        random_seed: int = 42
    ):
        np.random.seed(random_seed)
        # K-means has additional random state that is independent of np.random
        if isinstance(clusterer, KMeans):
            clusterer.random_state = random_seed

        self.data = np.array(data)
        self.n_clusters = clusterer.n_clusters

        # Shuffle the rows, just in case they are not randomly ordered in the input
        np.random.shuffle(self.data)

        self.scaler = scaler
        self.fit_scaler_on = fit_scaler_on

        self.clusterer = clusterer
        self.fit_clusterer_on = fit_clusterer_on

        self._fit()
        logger.info('Created the clusters from fingerprints data.')

    def _fit(self):
        self._fit_transform()
        self._fit_cluster()

    def _fit_transform(self):
        self.scaler.fit(self.data[:self.fit_scaler_on, :])

    def _fit_cluster(self):
        transformed = self._transform(self.data[:self.fit_clusterer_on, :])
        self.clusterer.fit(transformed)

    def _transform(self, data: np.ndarray):
        return self.scaler.transform(data)

    def predict(self, data: np.ndarray):
        transformed = self._transform(data)
        return self.clusterer.predict(transformed)


def inspect_clusters(
    clusterer: Union[ClustererFitter, Clusterer], data: np.ndarray, normalize: bool = True
):
    kmean_class = clusterer.predict(data)
    unique, counts = np.unique(kmean_class, return_counts=True)
    for u, c in zip(unique, counts):
        if normalize:
            c = 100 * c / data.shape[0]
        print(f'Cluster no {u:>2d}: {c:>6.2f} %')
