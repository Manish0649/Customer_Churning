"""Reusable evaluation, result tables and feature importance for classifiers."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from retention_strategy.retention_rules import assign_churn_risk


def churn_probabilities(pipeline, X):
    return pipeline.predict_proba(X)[:, list(pipeline.classes_).index(1)]


def evaluate_model(pipeline, X, y):
    probabilities = churn_probabilities(pipeline, X)
    predictions = pipeline.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
    }
    results = X.copy()
    results["actual_churn"] = np.asarray(y)
    results["predicted_churn"] = predictions
    results["churn_probability"] = probabilities
    return {
        "metrics": metrics,
        "classification_report": classification_report(y, predictions, zero_division=0, output_dict=True),
        "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
        "results": assign_churn_risk(results),
    }


def get_feature_importance(pipeline):
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.DataFrame({
        "feature": pipeline.named_steps["preprocessor"].get_feature_names_out(),
        "importance": importance,
    }).sort_values("importance", ascending=False)


def plot_model_results(evaluation, importance, title):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay(np.array(evaluation["confusion_matrix"]),
                           display_labels=["Stayed", "Churned"]).plot(cmap="Blues")
    plt.title(f"{title} - Confusion Matrix")
    if not importance.empty:
        importance.head(15).sort_values("importance").plot.barh(x="feature", y="importance")
        plt.title(f"{title} - Global Feature Importance")
        plt.tight_layout()
    plt.show()
