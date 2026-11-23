"""Phase D: model-selection rule U9 (approved by the supervisor). NO training here.

Inputs are ONLY development cross-validation results. The function has no way to
receive sealed-test results or accuracy, by design.

Definitions fixed BEFORE any training:
  * Standard error (SE): for the model with the highest mean F1 (label 1),
        SE = s / sqrt(5),   s = sample standard deviation (ddof = 1) of its five fold F1 values.
  * "Within one standard error": mean F1 >= best mean F1 - SE of the best model.
  * Step 2 tie-break: highest mean recall (label 1) among those models.
  * Tie = equal after rounding the mean recall to 3 decimals.
  * Step 3: simplicity order (simplest first), documented in docs/evaluation_protocol.md:
        LogisticRegression < DecisionTree < SVM < RandomForest < XGBoost
"""
from __future__ import annotations

import math

import numpy as np

APPROVED_MODELS = ["LogisticRegression", "DecisionTree", "SVM", "RandomForest", "XGBoost"]
SIMPLICITY_ORDER = list(APPROVED_MODELS)          # simplest first
N_FOLDS = 5


def standard_error(fold_scores) -> float:
    x = np.asarray(fold_scores, dtype=float)
    if len(x) != N_FOLDS:
        raise ValueError(f"expected {N_FOLDS} fold scores, got {len(x)}")
    return float(x.std(ddof=1) / math.sqrt(len(x)))


def select_model(cv_f1: dict, cv_recall: dict) -> dict:
    """cv_f1 / cv_recall: {model_name: [five fold values for label 1]}."""
    names = list(cv_f1)
    if set(names) != set(cv_recall):
        raise ValueError("F1 and recall results must cover the same models")
    unknown = set(names) - set(APPROVED_MODELS)
    if unknown:
        raise ValueError(f"only the five approved algorithms may be selected; got {sorted(unknown)}")
    mean_f1 = {m: float(np.mean(cv_f1[m])) for m in names}
    mean_rec = {m: float(np.mean(cv_recall[m])) for m in names}
    best = max(names, key=lambda m: mean_f1[m])
    se = standard_error(cv_f1[best])
    threshold = mean_f1[best] - se
    within = [m for m in names if mean_f1[m] >= threshold]
    trace = {"best_by_f1": best, "best_mean_f1": mean_f1[best], "se_of_best": se,
             "one_se_threshold": threshold, "within_one_se": within,
             "mean_f1": mean_f1, "mean_recall": mean_rec}
    if len(within) == 1:
        trace.update(selected=within[0], decided_by="step 1: highest mean F1 (no other model within one SE)")
        return trace
    top_rec = max(round(mean_rec[m], 3) for m in within)
    finalists = [m for m in within if round(mean_rec[m], 3) == top_rec]
    if len(finalists) == 1:
        trace.update(selected=finalists[0], decided_by="step 2: highest mean recall among models within one SE")
        return trace
    chosen = min(finalists, key=SIMPLICITY_ORDER.index)
    trace.update(selected=chosen, tied_on_recall=finalists,
                 decided_by="step 3: simplest model among those tied on recall")
    return trace
