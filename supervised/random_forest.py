from sklearn.ensemble import RandomForestClassifier
from .data_preprocessing import fit_pipeline


def train_random_forest(X, y, preprocessor=None):
    model = RandomForestClassifier(n_estimators=10, max_depth=12, min_samples_split=10,
        min_samples_leaf=5, random_state=42, n_jobs=-1, class_weight="balanced")
    return fit_pipeline(X, y, model, preprocessor)
