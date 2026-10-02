from sklearn.tree import DecisionTreeClassifier
from .data_preprocessing import fit_pipeline


def train_decision_tree(X, y, preprocessor=None):
    model = DecisionTreeClassifier(max_depth=10, min_samples_split=10,
        min_samples_leaf=5, random_state=42, class_weight="balanced")
    return fit_pipeline(X, y, model, preprocessor)
