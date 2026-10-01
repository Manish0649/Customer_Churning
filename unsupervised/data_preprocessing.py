import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

CLUSTERING_FEATURES = [
    'tenure',
    'monthlycharges',
    'totalcharges',
    'num_services',
    'customer_satisfaction',
    'num_complaints',
    'num_service_calls',
    'late_payments',
    'avg_monthly_gb',
    'days_since_last_interaction',
    'credit_score'
]


def extract_features(df, features=None):
    """
    Extract clustering features from dataframe.
    """
    if features is None:
        features = CLUSTERING_FEATURES
    return df[features].copy()


def fill_missing_values(df):
    """
    Fill missing values in clustering features using column median.
    """
    return df.fillna(df.median())


def scale_features(df, scaler=None):
    """
    Standardize clustering features using StandardScaler.
    """
    if scaler is None:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
    else:
        scaled_data = scaler.transform(df)
    return scaled_data, scaler


def get_sample(data, sample_size=10000, random_state=42):
    """
    Take a random sample from scaled array.
    """
    if sample_size >= data.shape[0]:
        indices = np.arange(data.shape[0])
        return data.copy(), indices

    rng = np.random.RandomState(random_state)
    sample_indices = rng.choice(data.shape[0], sample_size, replace=False)
    return data[sample_indices], sample_indices


def preprocess_data(df, features=None):
    """
    Complete preprocessing pipeline: Extract features, fill missing with median, scale.
    """
    cluster_df = extract_features(df, features)
    cluster_df_clean = fill_missing_values(cluster_df)
    cluster_scaled, scaler = scale_features(cluster_df_clean)
    return cluster_df_clean, cluster_scaled, scaler
