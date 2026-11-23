"""Phase D: does the five-feature model add anything beyond function SIZE? (development data only)

For each project-grouped validation fold this reports AUC and average precision of:
  * every single feature used as a raw score (direction chosen on the TRAINING rows of that fold);
  * Logistic Regression on token_count only, on the size trio (nloc, complexity, token_count),
    and on all five features (same pipeline settings as the approved model).
No sealed test data, no tuning, no model saving.

Run:  python -m src.ml.diagnose_v2
Output: reports/size_baseline_v2.json and reports/size_baseline_v2.md
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.ml import features, pipeline

F = features.FEATURE_NAMES
SUBSETS = {"token_count only": ["token_count"],
           "size trio (nloc, complexity, token_count)": ["nloc", "complexity", "token_count"],
           "all five features": F}


def _auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 and np.ptp(s) > 0 else 0.5


def raw_feature_scores(dev: pd.DataFrame) -> dict:
    """Direction (+1 or -1) is chosen on the training rows only, then applied to the validation fold."""
    out = {f: {"auc": [], "ap": [], "direction": []} for f in F}
    for k in sorted(dev.fold.unique()):
        tr, va = dev[dev.fold != k], dev[dev.fold == k]
        for f in F:
            d = 1 if _auc(tr.label, tr[f]) >= 0.5 else -1
            s = d * va[f].to_numpy(float)
            out[f]["auc"].append(_auc(va.label, s))
            out[f]["ap"].append(float(average_precision_score(va.label, s)))
            out[f]["direction"].append(d)
    return out


def lr_subsets(dev: pd.DataFrame, seed: int = 42) -> dict:
    y = dev.label.to_numpy(int)
    out = {}
    for name, cols in SUBSETS.items():
        X = dev[cols].to_numpy(float)
        aucs, aps = [], []
        for k in sorted(dev.fold.unique()):
            va = (dev.fold == k).to_numpy()
            m = pipeline.make_model("LogisticRegression", y[~va], seed).fit(X[~va], y[~va])
            s = pipeline.score(m, X[va])
            aucs.append(_auc(y[va], s)); aps.append(float(average_precision_score(y[va], s)))
        out[name] = {"auc": aucs, "ap": aps}
    return out


def _agg(v):
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v))), "folds": [float(x) for x in v]}


def diagnose(dev: pd.DataFrame) -> dict:
    features.check_features(dev)
    prev = float(np.mean([(g.label == 1).mean() for _, g in dev.groupby("fold")]))
    raw, lr = raw_feature_scores(dev), lr_subsets(dev)
    return {"mean_prevalence": prev,
            "raw_feature": {f: {"auc": _agg(v["auc"]), "ap": _agg(v["ap"]), "direction_by_fold": v["direction"]} for f, v in raw.items()},
            "logistic_regression_subsets": {n: {"auc": _agg(v["auc"]), "ap": _agg(v["ap"])} for n, v in lr.items()}}


def render(rep: dict) -> str:
    L = ["# Does the model add anything beyond function size? (development data only)", "",
         f"Random-scorer average precision = prevalence = {rep['mean_prevalence']:.3f}. Mean over 5 project-grouped folds; ± is the standard error.", "",
         "## Single feature as a raw score (direction fixed on training rows)", "", "| Feature | ROC-AUC | Average precision |", "|---|---|---|"]
    for f, v in rep["raw_feature"].items():
        L.append(f"| {f} | {v['auc']['mean']:.3f} ± {v['auc']['se']:.3f} | {v['ap']['mean']:.3f} ± {v['ap']['se']:.3f} |")
    L += ["", "## Logistic Regression on feature subsets (same settings as the approved model)", "",
          "| Features | ROC-AUC | Average precision |", "|---|---|---|"]
    for n, v in rep["logistic_regression_subsets"].items():
        L.append(f"| {n} | {v['auc']['mean']:.3f} ± {v['auc']['se']:.3f} | {v['ap']['mean']:.3f} ± {v['ap']['se']:.3f} |")
    L += ["", "Reading guide: if token_count alone is about as good as all five features, the models mainly "
              "recover function size, and Chapter Four must say so."]
    return "\n".join(L)


def main() -> int:
    from src.data import settings as S
    dev = features.load_development_v2(S.ROOT / "data" / "processed" / "functions_capped.csv",
                                       S.ROOT / "data" / "processed" / "split_assignment_v2.csv")
    rep = diagnose(dev)
    (S.REPORT_DIR / "size_baseline_v2.json").write_text(json.dumps(rep, indent=2))
    (S.REPORT_DIR / "size_baseline_v2.md").write_text(render(rep))
    print("Written: reports/size_baseline_v2.md and size_baseline_v2.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
