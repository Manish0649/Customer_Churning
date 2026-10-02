import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def apply_pca(data, n_components=None):
    """
    Apply PCA on scaled clustering data.
    """
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(data)
    return pca, pca_data


def get_variance_details(pca):
    """
    Calculate explained and cumulative variance ratios.
    """
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    return explained_variance, cumulative_variance


def plot_cumulative_variance(cumulative_variance):
    """
    Plot cumulative explained variance for PCA components.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(cumulative_variance) + 1),
        cumulative_variance * 100,
        marker='o'
    )
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance (%)")
    plt.title("PCA Cumulative Explained Variance")
    plt.axhline(y=90, linestyle='--', label='90% Variance', color='red')
    plt.legend()
    plt.show()


def plot_pca_clusters(pca_sample, cluster_labels, title="Customer Segments using PCA"):
    """
    Visualize customer segments in PCA 2D space.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(
        pca_sample[:, 0],
        pca_sample[:, 1],
        c=cluster_labels,
        s=10,
        alpha=0.5,
        cmap='viridis'
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(title)
    plt.colorbar(label="Cluster")
    plt.show()
