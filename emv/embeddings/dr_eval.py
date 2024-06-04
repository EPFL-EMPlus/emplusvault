import os
import random
import textwrap as tw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from umap import UMAP
from umap.umap_ import nearest_neighbors

# Metrics
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist
from coranking import coranking_matrix
from coranking.metrics import trustworthiness, continuity, LCMC
import mantel


# Embeddings helper functions
def normalize_embedding(points):
    dimensions = points.shape[1]

    ranges = []
    mins = []
    for d in range(dimensions):
        min_d = min(points[:, d])
        max_d = max(points[:, d])
        mins.append(min_d)
        ranges.append(max_d - min_d)

    normalized_points = []
    for point in points:
        normalized_point = []
        for d in range(dimensions):
            normalized_point.append((point[d] - mins[d]) / ranges[d])
        normalized_points.append(normalized_point)

    return np.array(normalized_points)


def compute_embeddings(features, reducer, params, normalize=True):
    embeddings = reducer(**params).fit_transform(features)
    if normalize:
        embeddings = normalize_embedding(embeddings)

    return {
        "reducer": reducer,
        "reducer_params": params,
        "embeddings": embeddings
    }


def compute_umap_embeddings(features, n_neighbors, min_dist=0.01, metric="cosine"):
    knn = nearest_neighbors(features,
                            n_neighbors=np.max(n_neighbors),
                            metric=metric,
                            metric_kwds={},
                            angular=False,
                            random_state=None)
    umap_embeddings = []
    for n in n_neighbors:
        embeddings_results = compute_embeddings(features, UMAP, {
                                                "n_neighbors": n, "min_dist": min_dist, "metric": metric, "precomputed_knn": knn})
        umap_embeddings.append(embeddings_results)

    return umap_embeddings


def format_params(params):
    return ", ".join([f"{k}={v}" for k, v in params.items() if k != "precomputed_knn"])


def plot_embeddings(embeddings_results, fig_title, d=4):
    n_plots = len(embeddings_results)
    n_cols = 4
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols,
                            figsize=(n_cols * d, n_rows * d))
    axs = axs.flatten()
    for i, result in enumerate(embeddings_results):
        coords = result["embeddings"]
        reducer = result["reducer"]
        params = result["reducer_params"]
        axs[i].scatter(coords[:, 0], coords[:, 1], s=0.1)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        title = f"{reducer.__name__} - params: {format_params(params)}"
        axs[i].set_title(tw.fill(title, width=40), fontsize=10)
    [axs[i].axis("off") for i in range(n_plots, n_rows * n_cols)]
    plt.suptitle(fig_title)
    plt.tight_layout()
    plt.show()


def plot_embeddings_with_images(embeddings, thumbnails, zoom=0.1, figsize=15):
    plt.figure(figsize=(figsize, figsize))  # Adjust the figure size if needed

    ax = plt.gca()
    for coords, img in zip(embeddings, thumbnails):
        # Adjust the zoom parameter to reduce the image size
        imagebox = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(imagebox, (coords[0], coords[1]), frameon=False)
        ax.add_artist(ab)

    plt.axis("off")
    plt.show()

# Co-Ranking Metrics


def compute_coranking_metrics(data_high_dim, data_low_dim, ks):
    """
    Compute trustworthiness and continuity metrics for a range of k values.

    Args:
        data_high_dim: original data in the high-dimensional space
        data_low_dim: data in the low-dimensional space
        ks: list of k values to compute the metrics on (i.e. how many neighbors to consider for the metrics)

    Returns:
        tuple of lists: trustworthiness and continuity values for each k value
    """
    Q = coranking_matrix(data_high_dim, data_low_dim)
    t_values = []
    c_values = []
    for k in ks:
        t_values.append(trustworthiness(Q, min_k=k, max_k=k + 1)[0])
        c_values.append(continuity(Q, min_k=k, max_k=k + 1)[0])

    return t_values, c_values

# Random Triplet Accuracy


def get_triplets_random(n_features, n_triplets=1000):
    triplets = [np.random.randint(0, n_features, 3) for _ in range(n_triplets)]
    return triplets


def get_triplets(knn, sampling, n_triplets=1000, n_neighbors=10):
    initial_points = np.random.randint(0, knn.shape[0], n_triplets)
    triplets = []
    for i in initial_points:
        if sampling == "local":  # Sample both j and k from the neighborhood of i
            j, k = np.random.choice(knn[i, 1:n_neighbors], 2, replace=False)
        elif sampling == "mixed":  # Sample j from the neighborhood of i and k from outside the neighborhood of i
            j = np.random.choice(knn[i, 1:n_neighbors])
            k = np.random.choice(knn[i, n_neighbors:])
        elif sampling == "global":  # Sample both j and k from outside the neighborhood of i
            j, k = np.random.choice(knn[i, n_neighbors:], 2, replace=False)
        else:
            raise ValueError(
                "Invalid sampling method. Choose between 'local', 'mixed' or 'global'.")

        triplets.append((i, j, k))
    return triplets


def compute_relative_distances(original_d, embeddings_d, triplets):
    relative_d_original = [
        np.sign(original_d[i, j] - original_d[i, k]) for i, j, k in triplets]
    relative_d_embedded = [
        np.sign(embeddings_d[i, j] - embeddings_d[i, k]) for i, j, k in triplets]

    return np.array(relative_d_original), np.array(relative_d_embedded)


def random_triplet_accuracy(knn_high_dim, original_d, embeddings_d, sampling, n_triplets=1000, n_repetitions=10, size_neighborhood=10):
    """
    Compute the Random Triplet Accuracy metric, which measures the proportion of triplets 
    for which the relative distances are preserved in the embedding compared to the original data.

    Args:
        data_high_dim: original data in the high-dimensional space
        embeddings_d: pairwise distances of the data in the low-dimensional space
        sampling: method to sample the triplets (local, mixed, global)
        n_triplets (int, optional): number of triplets to use. Defaults to 1000.
        n_repetitions (int, optional): number of repetitions to perform. Defaults to 10.
        size_neighborhood (int, optional): size of the neighborhood to consider. Defaults to 10.

    Returns:
        tuple: mean and standard deviation of the Random Triplet Accuracy
    """
    accs = []
    for _ in range(n_repetitions):
        triplets = get_triplets(knn_high_dim, n_triplets=n_triplets,
                                sampling=sampling, n_neighbors=size_neighborhood)
        relative_d_original, relative_d_embedded = compute_relative_distances(
            original_d, embeddings_d, triplets)
        acc = np.mean(relative_d_original == relative_d_embedded)
        accs.append(acc)

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)

    return mean_acc, std_acc

# Pearson Correlation Coefficient


def compute_pcc(data_high_dim, data_low_dim, n_clusters=100, n_sample=10000, n_repetitions=100):
    """
    Compute the Pearson Correlation Coefficient between the pairwise distances of the cluster centers in the high-dimensional and low-dimensional spaces.

    Args:
        data_high_dim: original data in the high-dimensional space
        data_low_dim: data in the low-dimensional space
        n_clusters (int, optional): Number of clusters for KMeans (should be fairly high). Defaults to 100.
        n_sample (int, optional): Size of the sample to consider (necessary to sample of larger datasets). Defaults to 10000.
        n_repetitions (int, optional): number of repetitions to perform. Defaults to 100.

    Returns:
        _type_: _description_
    """
    pccs = []
    if n_sample > len(data_high_dim):
        n_sample = len(data_high_dim)
    for _ in range(n_repetitions):
        sample_idx = np.random.choice(
            len(data_high_dim), n_sample, replace=False)
        kmeans_high_dim = KMeans(n_clusters=n_clusters).fit(
            data_high_dim[sample_idx])
        kmeans_low_dim = KMeans(n_clusters=n_clusters).fit(
            data_low_dim[sample_idx])

        clusters_centers_high_dim = kmeans_high_dim.cluster_centers_
        clusters_centers_low_dim = kmeans_low_dim.cluster_centers_

        clusters_d_high = pdist(clusters_centers_high_dim, metric="euclidean")
        clusters_d_low = pdist(clusters_centers_low_dim, metric="euclidean")

        pcc = mantel.test(clusters_d_high, clusters_d_low,
                          perms=1000, method="pearson")
        pccs.append(pcc.r)
    return np.mean(pccs), np.std(pccs)

# Global Score


def global_score(data_high_dim, data_low_dim):
    """
    Compute the global score metric, which measures the quality of the embedding by comparing the reconstruction error of the original data and the low-dimensional data.
    From https://github.com/eamid/trimap/blob/master/trimap/trimap_.py

    Args:
        data_high_dim: original data in the high-dimensional space
        data_low_dim: data in the low-dimensional space

    Returns:
        float: global score value
    """

    def global_loss_(X, Y):
        X = X - np.mean(X, axis=0)
        Y = Y - np.mean(Y, axis=0)
        A = X.T @ (Y @ np.linalg.inv(Y.T @ Y))
        return np.mean(np.power(X.T - A @ Y.T, 2))

    n_dims = data_low_dim.shape[1]
    data_low_dim_pca = PCA(n_components=n_dims).fit_transform(data_high_dim)
    gs_pca = global_loss_(data_high_dim, data_low_dim_pca)
    gs_emb = global_loss_(data_high_dim, data_low_dim)
    gs_score = np.exp(-(gs_emb - gs_pca) / gs_pca)
    return gs_score
