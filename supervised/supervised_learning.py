"""Supervised workflow, organized like unsupervised/unsupervised_learning.py."""
from time import perf_counter

import pandas as pd
from sklearn.model_selection import train_test_split

from .data_preprocessing import prepare_features_and_target
from .decision_tree import train_decision_tree
from .logistic_regression import train_logistic_regression
from .model_evaluation import evaluate_model, get_feature_importance, plot_model_results
from .random_forest import train_random_forest
from .svm_model import train_svm
from .xgboost_model import train_xgboost

TRAINERS = {
    "logistic_regression": train_logistic_regression,
    "decision_tree": train_decision_tree,
    "random_forest": train_random_forest,
    "svm": train_svm,
    "xgboost": train_xgboost,
}


def train_model(name, X, y):
    if name not in TRAINERS:
        raise ValueError(f"Unknown model {name!r}; choose from {list(TRAINERS)}")
    return TRAINERS[name](X, y)


def run_supervised_analysis(df, show_plots=False, models=None):
    """Return fitted pipelines, holdout results and comparison on one shared split.

    SVM uses at most 2,000 training records. Other models use all supplied rows.
    This is an experiment comparison, not a separate validation-based model selection.
    """
    names = list(TRAINERS) if models is None else list(models)
    if not names or any(name not in TRAINERS for name in names):
        raise ValueError(f"models must contain names from {list(TRAINERS)}")
    X, y = prepare_features_and_target(df)
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42)
    outputs, comparison = {}, []
    for name in names:
        started = perf_counter()
        pipeline = train_model(name, X_train, y_train)
        training_seconds = perf_counter() - started
        started = perf_counter()
        evaluation = evaluate_model(pipeline, X_holdout, y_holdout)
        evaluation_seconds = perf_counter() - started
        importance = get_feature_importance(pipeline)
        outputs[name] = {"pipeline": pipeline, **evaluation, "feature_importance": importance}
        comparison.append({"model": name, **evaluation["metrics"],
                           "training_seconds": training_seconds,
                           "evaluation_seconds": evaluation_seconds})
        if show_plots:
            plot_model_results(evaluation, importance, name)
    return {"models": outputs, "comparison": pd.DataFrame(comparison)}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="dataset/customer_churn_1M.csv")
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--models", nargs="+", choices=list(TRAINERS), default=list(TRAINERS))
    parser.add_argument("--show-plots", action="store_true")
    args = parser.parse_args()
    if args.sample_size < 40:
        parser.error("--sample-size must be at least 40")
    df = pd.read_csv(args.csv)
    if len(df) > args.sample_size:
        df, _ = train_test_split(df, train_size=args.sample_size, stratify=df.churn, random_state=42)
    output = run_supervised_analysis(df, args.show_plots, args.models)
    print(output["comparison"].to_string(index=False))


if __name__ == "__main__":
    main()
