"""Phase D: the five approved models with FIXED, pre-registered hyperparameters.
No hyperparameter tuning in this run (see docs/training_config.md). Scaling is part of the
scikit-learn Pipeline, so it is fitted on the training fold only. Class weights handle the mild
imbalance; no resampling and no threshold tuning in the first run."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
MODEL_NAMES = ["LogisticRegression", "DecisionTree", "SVM", "RandomForest", "XGBoost"]

HYPERPARAMETERS = {
    "LogisticRegression": dict(C=1.0, max_iter=1000, class_weight="balanced"),
    "DecisionTree": dict(max_depth=5, min_samples_leaf=20, class_weight="balanced"),
    "SVM": dict(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"),
    "RandomForest": dict(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                         class_weight="balanced_subsample"),
    "XGBoost": dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=1.0, min_child_weight=1, reg_lambda=1.0),
}


def _log():
    return FunctionTransformer(np.log1p, validate=False)   # fit-free, so it cannot leak


def make_model(name: str, y_train, seed: int = SEED, log_transform: bool = False):
    """Build an UNFITTED model. y_train is used only to set XGBoost's class-weight ratio.
    log_transform adds log1p before scaling for Logistic Regression and SVM (trees are unaffected)."""
    h = HYPERPARAMETERS.get(name)
    if h is None:
        raise ValueError(f"unknown model {name!r}; approved: {MODEL_NAMES}")
    if name == "LogisticRegression":
        steps = ([_log()] if log_transform else []) + [StandardScaler(), LogisticRegression(random_state=seed, **h)]
        return make_pipeline(*steps)
    if name == "SVM":
        steps = ([_log()] if log_transform else []) + [StandardScaler(), SVC(random_state=seed, **h)]
        return make_pipeline(*steps)
    if name == "DecisionTree":
        return DecisionTreeClassifier(random_state=seed, **h)
    if name == "RandomForest":
        return RandomForestClassifier(random_state=seed, n_jobs=1, **h)
    from xgboost import XGBClassifier
    y = np.asarray(y_train)
    spw = float((y == 0).sum() / max((y == 1).sum(), 1))
    return XGBClassifier(random_state=seed, n_jobs=1, tree_method="hist", eval_metric="logloss",
                         scale_pos_weight=spw, **h)


def score(model, X) -> np.ndarray:
    """Continuous score for label 1 (used for ROC-AUC). SVM uses its decision function."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)
