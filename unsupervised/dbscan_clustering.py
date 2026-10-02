import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from .data_preprocessing import CLUSTERING_FEATURES


def compute_k_distances(data, n_neighbors=11):
    """
    Find sorted distances to the k-th nearest neighbor for choosing eps.
    """
    neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    neighbors.fit(data)
    distances, _ = neighbors.kneighbors(data)
    k_distances = distances[:, n_neighbors - 1]
    return np.sort(k_distances)


def plot_k_distance_graph(k_distances, n_neighbors=11):
    """
    Plot k-distance graph for choosing optimal eps.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(k_distances)
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"Distance to {n_neighbors}th nearest neighbor")
    plt.title("K-Distance Graph for DBSCAN")
    plt.show()


def evaluate_dbscan_params(data, eps_values, min_samples_values=[11]):
    """
    Evaluate DBSCAN across multiple eps and min_samples values.
    """
    results = []
    for min_samples in min_samples_values:
        for eps in eps_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(data)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int(np.sum(labels == -1))
            noise_pct = (n_noise / len(labels)) * 100

            results.append({
                "eps": eps,
                "min_samples": min_samples,
                "clusters": n_clusters,
                "noise": n_noise,
                "noise_pct": round(noise_pct, 2)
            })
    return pd.DataFrame(results)


def fit_dbscan(data, eps=1.68, min_samples=5):
    """
    Fit DBSCAN with selected parameters.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_labels = dbscan.fit_predict(data)
    return dbscan, dbscan_labels


def profile_dbscan_noise(df, sample_indices, dbscan_labels, features=None, churn_col='churn'):
    """
    Profile noise vs normal customers and compare churn rates.
    """
    if features is None:
        features = CLUSTERING_FEATURES

    sub_df = df.iloc[sample_indices].copy()
    noise_mask = (dbscan_labels == -1)

    noise_pct = float((noise_mask).mean() * 100)

    dbscan_profile = pd.DataFrame({
        "Normal": sub_df.loc[~noise_mask, features].mean(),
        "Noise": sub_df.loc[noise_mask, features].mean()
    })

    noise_churn_rate = float(sub_df.loc[noise_mask, churn_col].mean() * 100) if churn_col in sub_df.columns else None
    normal_churn_rate = float(sub_df.loc[~noise_mask, churn_col].mean() * 100) if churn_col in sub_df.columns else None

    return {
        "noise_percentage": noise_pct,
        "feature_profile": dbscan_profile,
        "noise_churn_rate": noise_churn_rate,
        "normal_churn_rate": normal_churn_rate
    }


def plot_dbscan_pca(pca_sample, dbscan_labels):
    """
    Visualize DBSCAN clusters and noise points in PCA 2D space.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(
        pca_sample[:, 0],
        pca_sample[:, 1],
        c=dbscan_labels,
        s=10,
        alpha=0.5,
        cmap='tab10'
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("DBSCAN Customer Clusters and Noise")
    plt.colorbar(label="DBSCAN Cluster")
    plt.show()
