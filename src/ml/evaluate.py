"""Phase D: evaluation metrics and project-level bootstrap (approved U8). NO training here.

Label 1 = pre-fix method (positive class). Label 0 = post-fix method.
Accuracy is computed only as a supplementary figure; it is never used for selection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

METRICS = ["precision", "recall", "f1", "roc_auc", "average_precision", "recall_top10pct"]


def recall_at_top(y, s, frac: float = 0.10) -> float:
    """Share of label-1 rows found among the highest-scored `frac` of rows (review-prioritisation figure)."""
    y, s = np.asarray(y), np.asarray(s)
    k = max(1, int(np.ceil(frac * len(y))))
    idx = np.argsort(-s, kind="stable")[:k]
    return float(y[idx].sum() / max(y.sum(), 1))


def compute_metrics(y_true, y_pred, y_score) -> dict:
    """Positive class = label 1. Confusion matrix layout: [[TN, FP], [FN, TP]]."""
    y_true, y_pred, y_score = map(np.asarray, (y_true, y_pred, y_score))
    both = len(np.unique(y_true)) == 2
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if both else float("nan"),
        "average_precision": float(average_precision_score(y_true, y_score)) if both else float("nan"),
        "recall_top10pct": recall_at_top(y_true, y_score),
        "accuracy_supplementary": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": cm.tolist(),
        "n": int(len(y_true)),
    }


def bootstrap_ci_by_project(table: pd.DataFrame, n_boot: int = 1000, seed: int = 42,
                            alpha: float = 0.05) -> dict:
    """95% percentile CIs by resampling WHOLE projects with replacement.
    `table` needs columns: repo_url, y_true, y_pred, y_score."""
    rng = np.random.default_rng(seed)
    groups = {k: g[["y_true", "y_pred", "y_score"]].to_numpy() for k, g in table.groupby("repo_url")}
    keys = list(groups)
    draws = {m: [] for m in METRICS}
    skipped = 0
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), size=len(keys))
        arr = np.vstack([groups[keys[i]] for i in pick])
        if len(np.unique(arr[:, 0])) < 2:
            skipped += 1
            continue
        m = compute_metrics(arr[:, 0].astype(int), arr[:, 1].astype(int), arr[:, 2])
        for k in METRICS:
            draws[k].append(m[k])
    ci = {k: [float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2)))]
          for k, v in draws.items() if v}
    return {"ci95": ci, "n_boot": n_boot, "n_skipped_single_class": skipped,
            "n_projects_resampled": len(keys), "seed": seed,
            "point_estimate": compute_metrics(table.y_true.astype(int), table.y_pred.astype(int), table.y_score)}
