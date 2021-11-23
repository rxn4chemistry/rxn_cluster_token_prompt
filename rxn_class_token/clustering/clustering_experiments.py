# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.core.display import display, SVG
from rxn_chemutils.depict import smiles_depict_url
from sklearn.cluster import KMeans, Birch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, normalize

from rxn_class_token.clustering.data_loading import (
    load_schneider_df, PISTACHIO_FP_COLUMN, TPL_FP_COLUMN, RXN_SMILES_COLUMN, load_yyy_df,
    load_zzz_df
)

# %matplotlib inline

# ### Selecting which fp to use
from rxn_class_token.clustering.clusterer import ClustererFitter, inspect_clusters

selected_fp_column = PISTACHIO_FP_COLUMN
print(f'Available fingerprints (Pistachio: "{PISTACHIO_FP_COLUMN}", TPL 1k: "{TPL_FP_COLUMN}")')
print('Selected:', selected_fp_column)

# ### Loading Schneider dataset

# + tags=[]
schneider_df = load_schneider_df()
fps = np.array(schneider_df[selected_fp_column].tolist())
print(fps.shape)
schneider_df.head()
# -

# ### TSNE plot on Schneider

tsne_subset = 5000
tsne_fps = fps[:tsne_subset, :]
tsne_projection = TSNE().fit_transform(tsne_fps)
plt.scatter(*tsne_projection.T)
plt.show()

labels_sub = schneider_df['rxn_Class'].tolist()[:tsne_subset]
color_palette = sns.color_palette('Paired', 48)
cluster_colors = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in labels_sub]
plt.scatter(*tsne_projection.T, s=50, linewidth=0, c=cluster_colors, alpha=0.25)
plt.show()

# ### Clustering functionality


def get_stats(
    clusterer: ClustererFitter, df: pd.DataFrame, data: np.ndarray, print_plots: bool = True
):
    df = df.copy()

    kmean_class = clusterer.predict(data)
    df['kmeans_class'] = kmean_class.tolist()

    grouped_df = df.groupby(['rxn_Class', 'kmeans_class']).size().unstack(fill_value=0)
    print(grouped_df.to_string())

    pred_matrix = grouped_df.to_numpy()

    if print_plots:
        _ = plt.subplots(figsize=(10, 7))
        normalized_array = normalize(pred_matrix, norm="l1")
        plt.imshow(normalized_array)
        plt.show()

        _ = plt.subplots(figsize=(10, 7))
        normalized_array = normalize(pred_matrix, norm="l1", axis=0)
        plt.imshow(normalized_array)
        plt.show()


def show_in_tsne(clusterer):
    labels_sub = clusterer.predict(tsne_fps)
    color_palette = sns.color_palette('Paired', 48)
    cluster_colors = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in labels_sub]
    plt.scatter(*tsne_projection.T, s=50, linewidth=0, c=cluster_colors, alpha=0.25)
    plt.show()


# ### Tests - try out different clustering approaches

clusterer = ClustererFitter(data=fps, scaler=PCA(n_components=30), clusterer=KMeans(n_clusters=30))
get_stats(clusterer, schneider_df, fps, print_plots=False)
show_in_tsne(clusterer)

clusterer = ClustererFitter(data=fps, scaler=PCA(n_components=10), clusterer=KMeans(n_clusters=20))
get_stats(clusterer, schneider_df, fps, print_plots=False)
show_in_tsne(clusterer)

clusterer = ClustererFitter(data=fps, scaler=StandardScaler(), clusterer=KMeans(n_clusters=15))
get_stats(clusterer, schneider_df, fps, print_plots=False)
show_in_tsne(clusterer)

clusterer = ClustererFitter(data=fps, scaler=StandardScaler(), clusterer=Birch(n_clusters=15))
get_stats(clusterer, schneider_df, fps, print_plots=False)
show_in_tsne(clusterer)

# ### Optimal PCA dimension
#
# https://medium.com/@dmitriy.kavyazin/principal-component-analysis-and-k-means-clustering-to-visualize-a-high-dimensional-dataset-577b2a7a5fe2

# Standardize the data to have a mean of ~0 and a variance of 1
X_std = StandardScaler().fit_transform(fps)
# Create a PCA instance: pca
pca = PCA(n_components=20)
principalComponents = pca.fit_transform(X_std)
# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)  # Save components to a DataFrame
PCA_components = pd.DataFrame(principalComponents)

# ### Find k-means elbow
#
# https://medium.com/@dmitriy.kavyazin/principal-component-analysis-and-k-means-clustering-to-visualize-a-high-dimensional-dataset-577b2a7a5fe2


def elbow_curve(
    components_df: pd.DataFrame, n_components: int, plot_from: int = 1, plot_until: int = 15
):
    ks = range(plot_from, plot_until)

    inertias = []
    for k in ks:
        # Create a KMeans instance with k clusters: model
        model = KMeans(n_clusters=k)

        # Fit model to samples
        model.fit(components_df.iloc[:, :n_components])

        # Append the inertia to the list of inertias
        inertias.append(model.inertia_)

    plt.plot(ks, inertias, '-o', color='black')
    plt.xlabel('number of clusters, k')
    plt.ylabel('inertia')
    plt.xticks(ks)
    plt.title(f'K-means inertia, {n_components} components of PCA')
    plt.show()


elbow_curve(PCA_components, n_components=3)

elbow_curve(PCA_components, n_components=3, plot_from=15, plot_until=30)

elbow_curve(PCA_components, n_components=5)

elbow_curve(PCA_components, n_components=9)

# ### Selected clustering for Schneider only

selected_n_components = 3
selected_n_clusters = 8

clusterer = ClustererFitter(
    data=fps,
    scaler=PCA(n_components=selected_n_components),
    clusterer=KMeans(n_clusters=selected_n_clusters)
)
get_stats(clusterer, schneider_df, fps, print_plots=False)
show_in_tsne(clusterer)

# ### Reaction analysis


def render_smiles(smiles: str):
    try:
        print(smiles)
        display(SVG(url=smiles_depict_url(smiles)))
    except Exception:
        print('[INVALID]')


n_show = 5
schneider_df['kmeans_class'] = clusterer.predict(fps)
for i in range(clusterer.n_clusters):
    print('Cluster no', i)
    rxn_smiles = schneider_df.loc[schneider_df['kmeans_class'] == i][RXN_SMILES_COLUMN].tolist()
    for rxn in random.sample(rxn_smiles, n_show):
        render_smiles(rxn)
    print('\n\n\n\n\n')

# ## Inspect clusters on YYY and ZZZ

# +
yyy_df = load_yyy_df()
yyy_fps = np.array(yyy_df[selected_fp_column].tolist())

zzz_df = load_zzz_df()
zzz_fps = np.array(zzz_df[selected_fp_column].tolist())

# -

inspect_clusters(clusterer, yyy_fps)

inspect_clusters(clusterer, zzz_fps)

# + tags=[]
inspect_clusters(clusterer, fps)
# -

# ### Clustering on all data at the same time

all_fps = np.concatenate((fps, yyy_fps, zzz_fps), axis=0)
print(all_fps.shape)

np.random.shuffle(all_fps)
all_fps_subset = all_fps[:50000, :]
print(all_fps_subset.shape)

# PCA

# Standardize the data to have a mean of ~0 and a variance of 1
X_std = StandardScaler().fit_transform(all_fps_subset)
# Create a PCA instance: pca
pca = PCA(n_components=20)
principalComponents = pca.fit_transform(X_std)
# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)  # Save components to a DataFrame
PCA_components = pd.DataFrame(principalComponents)

elbow_curve(PCA_components, n_components=3)

clusterer = ClustererFitter(
    data=all_fps_subset,
    scaler=PCA(n_components=3),
    clusterer=KMeans(n_clusters=8),
    random_seed=22
)

inspect_clusters(clusterer, yyy_fps)

inspect_clusters(clusterer, zzz_fps)

# + tags=[]
inspect_clusters(clusterer, fps)
