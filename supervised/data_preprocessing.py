"""Shared supervised preprocessing, delegated to the team's data pipeline."""
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from data_processing.data_processing import (
    build_preprocessor, engineer_features, load_data, prepare_features_and_target,
)


def fit_pipeline(X, y, model, preprocessor=None):
    """Fit a fresh transformer per model; never share fitted mutable state."""
    transformer = build_preprocessor(X) if preprocessor is None else clone(preprocessor)
    return Pipeline([("preprocessor", transformer), ("model", model)]).fit(X, y)


__all__ = ["build_preprocessor", "engineer_features", "load_data",
           "prepare_features_and_target", "fit_pipeline"]
