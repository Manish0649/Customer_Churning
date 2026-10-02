from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from .data_preprocessing import fit_pipeline


def train_svm(X, y, preprocessor=None, sample_size=2000):
    """Bound expensive probability-enabled RBF fitting; allow small datasets."""
    if sample_size < 4:
        raise ValueError("SVM sample_size must be at least 4")
    if len(X) > sample_size:
        X, _, y, _ = train_test_split(X, y, train_size=sample_size,
                                      stratify=y, random_state=42)
    model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                class_weight="balanced", random_state=42)
    return fit_pipeline(X, y, model, preprocessor)
