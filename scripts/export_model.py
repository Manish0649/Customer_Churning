"""Export a serving artifact without changing the team's research scripts.

Run from the repository root: python -m scripts.export_model
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from supervised.data_preprocessing import prepare_features_and_target
from supervised.supervised_learning import TRAINERS, train_model
from supervised.model_evaluation import evaluate_model, get_feature_importance
from unsupervised.data_preprocessing import CLUSTERING_FEATURES


def export_model(csv_path: Path, output: Path, sample_size: int, model_name: str):
    df = pd.read_csv(csv_path)
    dataset_rows = len(df)
    if len(df) < 40 or df.churn.isna().any() or not df.churn.isin([0, 1]).all():
        raise ValueError("Need at least 40 rows with binary, non-null churn labels")
    if df.churn.nunique() != 2 or df.churn.value_counts().min() < 4:
        raise ValueError("Need both churn classes with at least four examples each")
    if sample_size < len(df):
        df, _ = train_test_split(df, train_size=sample_size, stratify=df.churn, random_state=42)
    train, holdout = train_test_split(df, test_size=0.2, stratify=df.churn, random_state=42)
    X, y = prepare_features_and_target(train)
    X_holdout, y_holdout = prepare_features_and_target(holdout)
    if X.isna().all().any():
        raise ValueError("Training sample contains an entirely missing feature; use a larger sample")
    pipeline = train_model(model_name, X, y)
    metrics = evaluate_model(pipeline, X_holdout, y_holdout)["metrics"]
    segmentation = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", KMeans(n_clusters=4, random_state=42, n_init=10)),
    ])
    labels = segmentation.fit_predict(train[CLUSTERING_FEATURES])
    profile_frame = train[CLUSTERING_FEATURES].copy()
    profile_frame["segment"] = labels
    profiles = {
        str(int(segment)): {key: float(value) for key, value in row.items()}
        for segment, row in profile_frame.groupby("segment").mean().iterrows()
    }
    importance = get_feature_importance(pipeline).head(20)
    feature_importance = dict(zip(importance.feature, map(float, importance.importance)))
    thresholds = {"low": float(train.totalcharges.quantile(0.33)),
                  "high": float(train.totalcharges.quantile(0.67))}
    classifier_rows = min(len(train), 2000) if model_name == "svm" else len(train)
    metadata = {
        "artifact_version": 1, "model_name": model_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": dataset_rows, "sample_rows": len(df),
        "training_rows": classifier_rows, "holdout_rows": len(holdout),
        "metrics": metrics, "feature_importance": feature_importance,
        "input_features": list(X.columns), "value_thresholds": thresholds,
        "segment_profiles": profiles,
        "note": "Configured serving model, not a model-comparison winner. Metrics use a held-out "
                "20% split of a stratified sample (seed 42). Importance is global; logistic "
                "importance uses absolute encoded-feature coefficients. Segment IDs are K-Means "
                "groups, not validated business categories. Rewards and ROI are rule-based scenarios. "
                "SVM alone fits at most 2,000 training records; segmentation and value thresholds "
                "use the full training split.",
    }
    artifact = {
        "metadata": metadata, "sklearn_version": sklearn.__version__,
        "pipeline": pipeline, "segmentation": segmentation,
        "numeric_features": list(X.select_dtypes(include="number").columns),
        "clustering_features": CLUSTERING_FEATURES,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    joblib.dump(artifact, temporary)
    temporary.replace(output)
    print(f"Saved {model_name} to {output}; sample={len(df)}, training={len(train)}")
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("dataset/customer_churn_1M.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/churn_bundle.joblib"))
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--model", choices=list(TRAINERS),
                        default="random_forest")
    args = parser.parse_args()
    if args.sample_size < 40:
        parser.error("--sample-size must be at least 40")
    export_model(args.csv, args.output, args.sample_size, args.model)
