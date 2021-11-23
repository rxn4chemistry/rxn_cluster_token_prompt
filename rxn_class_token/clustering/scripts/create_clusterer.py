import pickle

import click
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from ..clusterer import ClustererFitter, inspect_clusters, Clusterer
from ..data_loading import load_yyy_df, load_zzz_df, PISTACHIO_FP_COLUMN, load_schneider_df


@click.command()
@click.option('--clusterer_pkl', type=str, required=True, help='Where to store the clusterer.')
def main(clusterer_pkl: str):
    """Create a clusterer based on a mix of Schneider and XXX data.

    The clusterer is used later on to get the reaction class for the diversity
    model relying on class tokens."""

    selected_fp_column = PISTACHIO_FP_COLUMN

    # Load Schneider (50k)
    schneider_df = load_schneider_df()
    schneider_fps = np.array(schneider_df[selected_fp_column].tolist())
    print('Loaded Schneider:', len(schneider_fps))

    # Load YYY (50k)
    yyy_df = load_yyy_df()
    yyy_fps = np.array(yyy_df[selected_fp_column].tolist())
    print('Loaded YYY:', len(yyy_fps))

    # Load ZZZ (~15k)
    zzz_df = load_zzz_df()
    zzz_fps = np.array(zzz_df[selected_fp_column].tolist())
    print('Loaded ZZZ:', len(zzz_fps))

    all_fps = np.concatenate((schneider_fps, yyy_fps, zzz_fps), axis=0)
    print('Merged, shuffled:', len(all_fps))

    pca = PCA(n_components=3)
    kmeans = KMeans(n_clusters=8)

    print('Fitting clusterer...')
    _ = ClustererFitter(
        data=all_fps,
        scaler=pca,
        clusterer=kmeans,
        random_seed=42,
        fit_scaler_on=len(all_fps),
        fit_clusterer_on=len(all_fps),
    )
    print('Fitting clusterer... Done.')

    clusterer = Clusterer(pca=pca, kmeans=kmeans)

    print('Clusters on YYY')
    inspect_clusters(clusterer, yyy_fps)
    print('Clusters on ZZZ')
    inspect_clusters(clusterer, zzz_fps)
    print('Clusters on Schneider')
    inspect_clusters(clusterer, schneider_fps)

    print(f'Saving clusterer to {clusterer_pkl}...')
    with open(clusterer_pkl, 'wb') as f:
        pickle.dump(clusterer, f)
    print(f'Saving clusterer to {clusterer_pkl}... Done')

    print('\n\nCheck: reloading the clusterer, should print exact same values as above')
    with open(clusterer_pkl, 'rb') as f:
        loaded: Clusterer = pickle.load(f)
    print('Loaded on YYY')
    inspect_clusters(loaded, yyy_fps)
    print('Loaded on ZZZ')
    inspect_clusters(loaded, zzz_fps)
    print('Loaded on Schneider')
    inspect_clusters(loaded, schneider_fps)


if __name__ == '__main__':
    main()
