import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from .data_preprocessing import CLUSTERING_FEATURES


def calculate_inertias(data, k_range=range(2, 11), random_state=42):
    """
    Calculate inertias for elbow method across range of K values.
    """
    inertias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    return inertias


def plot_elbow_curve(k_values, inertias):
    """
    Plot elbow curve for optimal K selection.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, marker='o')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal K")
    plt.show()


def calculate_silhouette_scores(data, k_range=range(2, 9), random_state=42):
    """
    Calculate silhouette scores across range of K values.
    """
    scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        scores.append(score)
    return scores


def plot_silhouette_scores(k_values, silhouette_scores):
    """
    Plot silhouette scores across K values.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Analysis for Optimal K")
    plt.show()


def train_kmeans(data, n_clusters=4, random_state=42):
    """
    Train K-Means model on dataset.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(data)
    return kmeans, cluster_labels


def profile_clusters(df, cluster_labels, features=None):
    """
    Profile each customer segment with feature averages.
    """
    if features is None:
        features = CLUSTERING_FEATURES
    df_copy = df.copy()
    df_copy['cluster'] = cluster_labels
    return df_copy.groupby('cluster')[features].mean()


def compare_cluster_churn(df, cluster_labels, churn_col='churn'):
    """
    Compare churn rate percentage across customer segments.
    """
    df_copy = df.copy()
    df_copy['cluster'] = cluster_labels
    return df_copy.groupby('cluster')[churn_col].mean() * 100
