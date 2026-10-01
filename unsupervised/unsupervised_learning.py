
"""
Customer Churning - Unsupervised Learning Workflow
Includes K-Means Clustering, PCA, and DBSCAN Analysis.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from .data_preprocessing import CLUSTERING_FEATURES, preprocess_data, get_sample
from .kmeans_clustering import (
    calculate_inertias,
    plot_elbow_curve,
    calculate_silhouette_scores,
    plot_silhouette_scores,
    train_kmeans,
    profile_clusters,
    compare_cluster_churn
)
from .pca_analysis import (
    apply_pca,
    get_variance_details,
    plot_cumulative_variance,
    plot_pca_clusters
)
from .dbscan_clustering import (
    compute_k_distances,
    plot_k_distance_graph,
    evaluate_dbscan_params,
    fit_dbscan,
    profile_dbscan_noise,
    plot_dbscan_pca
)


def run_unsupervised_analysis(df, show_plots=False):
    """
    Run complete unsupervised learning workflow on customer dataset.

    Args:
        df (pd.DataFrame): Customer DataFrame.
        show_plots (bool): Whether to display matplotlib plots.

    Returns:
        dict: Processed DataFrame, models, and summary profiles.
    """
    print("--- 1. Preprocessing & Feature Scaling ---")
    cluster_df_clean, cluster_scaled, scaler = preprocess_data(df)
    print("Features shape:", cluster_scaled.shape)

    # Sampling for K selection
    cluster_sample, _ = get_sample(cluster_scaled, sample_size=100000, random_state=42)

    print("--- 2. K-Means Elbow & Silhouette Analysis ---")
    inertias = calculate_inertias(cluster_sample, k_range=range(2, 11))
    if show_plots:
        plot_elbow_curve(range(2, 11), inertias)

    silhouette_sample, _ = get_sample(cluster_sample, sample_size=10000, random_state=42)
    silhouette_scores = calculate_silhouette_scores(silhouette_sample, k_range=range(2, 9))
    if show_plots:
        plot_silhouette_scores(range(2, 9), silhouette_scores)

    print("--- 3. Train K-Means (k=4) ---")
    kmeans, cluster_labels = train_kmeans(cluster_scaled, n_clusters=4, random_state=42)
    df_result = df.copy()
    df_result['cluster'] = cluster_labels

    cluster_profile = profile_clusters(df_result, cluster_labels)
    print("\nCluster Profile:")
    print(cluster_profile)

    if 'churn' in df_result.columns:
        cluster_churn = compare_cluster_churn(df_result, cluster_labels)
        print("\nCluster Churn Rate (%):")
        print(cluster_churn)

    print("\n--- 4. PCA (Principal Component Analysis) ---")
    pca, pca_data = apply_pca(cluster_scaled)
    explained_var, cum_var = get_variance_details(pca)

    for i, var in enumerate(explained_var, start=1):
        print(f"PC{i}: {var:.4f} ({var*100:.2f}%)")

    if show_plots:
        plot_cumulative_variance(cum_var)

    pca_sample, sample_indices = get_sample(pca_data, sample_size=10000, random_state=42)
    pca_sample_labels = cluster_labels[sample_indices]
    if show_plots:
        plot_pca_clusters(pca_sample, pca_sample_labels)

    print("\n--- 5. DBSCAN Clustering & Outlier Detection ---")
    dbscan_sample, dbscan_sample_indices = get_sample(cluster_scaled, sample_size=10000, random_state=42)

    k_distances = compute_k_distances(dbscan_sample, n_neighbors=11)
    if show_plots:
        plot_k_distance_graph(k_distances, n_neighbors=11)

    dbscan, dbscan_labels = fit_dbscan(dbscan_sample, eps=1.68, min_samples=5)
    noise_results = profile_dbscan_noise(
        df_result,
        dbscan_sample_indices,
        dbscan_labels
    )

    print(f"Noise percentage: {noise_results['noise_percentage']:.2f}%")
    if noise_results['noise_churn_rate'] is not None:
        print(f"Noise customer churn rate: {noise_results['noise_churn_rate']:.2f}%")
        print(f"Normal customer churn rate: {noise_results['normal_churn_rate']:.2f}%")

    if show_plots:
        dbscan_pca_sample = pca_data[dbscan_sample_indices]
        plot_dbscan_pca(dbscan_pca_sample, dbscan_labels)

    return {
        "df": df_result,
        "kmeans": kmeans,
        "cluster_profile": cluster_profile,
        "pca": pca,
        "pca_data": pca_data,
        "dbscan": dbscan,
        "dbscan_labels": dbscan_labels,
        "noise_results": noise_results
    }
