from .data_preprocessing import fit_pipeline


def train_xgboost(X, y, preprocessor=None):
    # Lazy import keeps the other four models usable without XGBoost installed.
    from xgboost import XGBClassifier
    model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        eval_metric="logloss")
    return fit_pipeline(X, y, model, preprocessor)
