from sklearn.linear_model import LogisticRegression
from .data_preprocessing import fit_pipeline


def train_logistic_regression(X, y, preprocessor=None):
    return fit_pipeline(X, y, LogisticRegression(max_iter=1000), preprocessor)
