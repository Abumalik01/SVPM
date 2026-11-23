"""Phase D: signal diagnosis on DEVELOPMENT data only (no sealed test data, no model saving).

Why: cross-validation gave ROC-AUC about 0.5 for all five models. This script tests WHY, per the
supervisor's instruction to investigate rather than substitute datasets. It reports:
  1. Trivial baselines per fold (always predict label 1) next to the real models.
  2. Within-pair differences of the five features (pre-fix vs post-fix of the same method).
  3. How much feature variance is within a pair (where the label signal lives) vs between methods.
  4. Single-feature AUC by fold, and for paired vs unpaired rows.
  5. The primary-dataset winner trained on primary training folds and evaluated on the
     pair-complete (secondary) validation rows, including a within-pair ranking accuracy.

Run:  python -m src.ml.diagnose            (reads reports/cv_results.json)
Output: reports/signal_diagnosis.json and reports/signal_diagnosis.md
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from src.ml import evaluate, features, pipeline

F = features.FEATURE_NAMES


def trivial_baselines(dev: pd.DataFrame) -> dict:
    rows = []
    for k, g in dev.groupby("fold"):
        p = float((g.label == 1).mean())
        rows.append({"fold": int(k), "prevalence": p, "always_positive_f1": 2 * p / (1 + p)})
    return {"folds": rows, "mean_prevalence": float(np.mean([r["prevalence"] for r in rows])),
            "mean_always_positive_f1": float(np.mean([r["always_positive_f1"] for r in rows]))}


def wide_pairs(dev: pd.DataFrame) -> pd.DataFrame:
    d = dev[dev.pair_complete].drop_duplicates(subset=["unit", "label"])
    w = d.pivot(index="unit", columns="label", values=F).dropna()
    return w          # columns: (feature, 0=post) and (feature, 1=pre)


def within_pair_differences(dev: pd.DataFrame) -> dict:
    w = wide_pairs(dev)
    out = {"n_pairs": int(len(w)), "features": {}}
    all_same = np.ones(len(w), dtype=bool)
    for f in F:
        pre, post = w[(f, 1)].to_numpy(float), w[(f, 0)].to_numpy(float)
        all_same &= (pre == post)
        diff = post - pre
        nz = diff != 0
        smaller = int((pre < post).sum())
        n_nz = int(nz.sum())
        p = float(binomtest(smaller, n_nz, 0.5).pvalue) if n_nz else None
        out["features"][f] = {"share_identical": float((~nz).mean()),
                              "share_pre_smaller": float((pre < post).mean()),
                              "share_pre_larger": float((pre > post).mean()),
                              "median_post_minus_pre": float(np.median(diff)),
                              "sign_test_p_value": p}
    out["share_all_five_identical"] = float(all_same.mean())
    return out


def variance_decomposition(dev: pd.DataFrame) -> dict:
    """On log1p scale: share of total variance that lies INSIDE a pair. A small share means
    the before/after change is tiny compared with the differences between different methods."""
    w = wide_pairs(dev)
    out = {}
    for f in F:
        pre, post = np.log1p(w[(f, 1)].to_numpy(float)), np.log1p(w[(f, 0)].to_numpy(float))
        total = float(np.var(np.concatenate([pre, post]), ddof=1))
        within = float(np.mean((pre - post) ** 2 / 2.0))
        out[f] = {"within_pair_variance_share": within / total if total > 0 else None}
    return out


def _auc(y, x):
    return float(roc_auc_score(y, x)) if len(np.unique(y)) == 2 and np.ptp(x) > 0 else 0.5


def univariate_auc(dev: pd.DataFrame) -> dict:
    """AUC of each raw feature as a score for label 1. Below 0.5: pre-fix methods tend to be smaller."""
    unit_n = dev.groupby("unit").label.transform("nunique")
    out = {}
    for f in F:
        out[f] = {"pooled": _auc(dev.label, dev[f]),
                  "by_fold": [_auc(g.label, g[f]) for _, g in dev.groupby("fold")],
                  "paired_rows_only": _auc(dev[unit_n == 2].label, dev[unit_n == 2][f]),
                  "unpaired_rows_only": _auc(dev[unit_n < 2].label, dev[unit_n < 2][f])}
    return out


def pairwise_accuracy(val: pd.DataFrame, scores: np.ndarray) -> float | None:
    v = val.assign(score=scores)
    v = v[v.pair_complete].drop_duplicates(subset=["unit", "label"])
    w = v.pivot(index="unit", columns="label", values="score").dropna()
    if not len(w):
        return None
    return float(((w[1] > w[0]).sum() + 0.5 * (w[1] == w[0]).sum()) / len(w))


def winner_on_pairs(dev: pd.DataFrame, model_name: str, seed: int = 42) -> dict:
    """Train on ALL primary training rows of each fold; evaluate only on the pair-complete
    validation rows of that fold. Secondary rows are a subset of primary rows, and folds are
    assigned by project, so no validation row or its pair partner is seen in training."""
    X, y = dev[F].to_numpy(float), dev.label.to_numpy(int)
    folds, pw = [], []
    for k in sorted(dev.fold.unique()):
        va = (dev.fold == k).to_numpy()
        m = pipeline.make_model(model_name, y[~va], seed).fit(X[~va], y[~va])
        sel = va & dev.pair_complete.to_numpy()
        s = pipeline.score(m, X[sel])
        folds.append(evaluate.compute_metrics(y[sel], m.predict(X[sel]), s))
        pw.append(pairwise_accuracy(dev[va], pipeline.score(m, X[va])))
    agg = {k: float(np.mean([f[k] for f in folds])) for k in ("precision", "recall", "f1", "roc_auc")}
    return {"model": model_name, "mean": agg, "folds": folds, "pairwise_ranking_accuracy_by_fold": pw,
            "pairwise_ranking_accuracy_mean": float(np.mean([p for p in pw if p is not None]))}


def diagnose(dev: pd.DataFrame, winner: str) -> dict:
    features.check_features(dev)
    return {"trivial": trivial_baselines(dev), "within_pair": within_pair_differences(dev),
            "variance": variance_decomposition(dev), "univariate_auc": univariate_auc(dev),
            "winner_on_pair_complete_rows": winner_on_pairs(dev, winner)}


def render(rep: dict, cv: dict | None = None) -> str:
    t, w = rep["trivial"], rep["within_pair"]
    L = ["# Signal diagnosis (development data only; sealed test not used)", "",
         "Label 1 = pre-fix method; label 0 = post-fix version. Not independently verified vulnerable/safe code.", "",
         f"## 1. Trivial baseline", "",
         f"Mean prevalence of label 1 in validation folds: {t['mean_prevalence']:.3f}. "
         f"A classifier that labels everything pre-fix gets F1 = {t['mean_always_positive_f1']:.3f}.", ""]
    if cv:
        L += ["| Model | Primary F1 | Predicted-positive rate | Precision minus prevalence |", "|---|---|---|---|"]
        for n, v in cv["primary"]["models"].items():
            fo = v["folds"]
            ppr = np.mean([(f["confusion_matrix"][0][1] + f["confusion_matrix"][1][1]) / f["n"] for f in fo])
            L.append(f"| {n} | {v['summary']['f1']['mean']:.3f} | {ppr:.2f} | {v['summary']['precision']['mean'] - t['mean_prevalence']:+.3f} |")
        L.append("")
    L += [f"## 2. Within-pair differences ({w['n_pairs']:,} complete pairs)", "",
          f"Share of pairs where all five features are identical: **{w['share_all_five_identical']:.1%}**", "",
          "| Feature | Identical | Pre-fix smaller | Pre-fix larger | Median (post - pre) | Sign-test p |", "|---|---|---|---|---|---|"]
    for f, v in w["features"].items():
        p = "n/a" if v["sign_test_p_value"] is None else f"{v['sign_test_p_value']:.3g}"
        L.append(f"| {f} | {v['share_identical']:.1%} | {v['share_pre_smaller']:.1%} | {v['share_pre_larger']:.1%} | {v['median_post_minus_pre']:.1f} | {p} |")
    L += ["", "## 3. Share of variance inside a pair (log scale)", "", "| Feature | Within-pair share |", "|---|---|"]
    L += [f"| {f} | {v['within_pair_variance_share']:.3f} |" for f, v in rep["variance"].items()]
    L += ["", "## 4. Single-feature AUC for label 1 (0.5 = no signal; below 0.5 = pre-fix tends to be smaller)", "",
          "| Feature | Pooled | Paired rows | Unpaired rows | Fold range |", "|---|---|---|---|---|"]
    for f, v in rep["univariate_auc"].items():
        L.append(f"| {f} | {v['pooled']:.3f} | {v['paired_rows_only']:.3f} | {v['unpaired_rows_only']:.3f} | {min(v['by_fold']):.3f} to {max(v['by_fold']):.3f} |")
    ww = rep["winner_on_pair_complete_rows"]
    L += ["", f"## 5. Primary winner ({ww['model']}) evaluated on pair-complete validation rows", "",
          json.dumps({k: round(v, 3) for k, v in ww["mean"].items()}),
          f"Within-pair ranking accuracy (does the model score the pre-fix version higher?): "
          f"{ww['pairwise_ranking_accuracy_mean']:.3f} (0.5 = chance).", ""]
    return "\n".join(L)


def main() -> int:
    from src.data import settings as S
    cv = json.loads((S.REPORT_DIR / "cv_results.json").read_text())
    p, _ = features.load_development(S.ROOT / "data" / "processed" / "methods_c_cpp.csv",
                                     S.ROOT / "data" / "processed" / "split_assignment.csv")
    rep = diagnose(p, cv["primary"]["selection"]["selected"])
    (S.REPORT_DIR / "signal_diagnosis.json").write_text(json.dumps(rep, indent=2, default=str))
    (S.REPORT_DIR / "signal_diagnosis.md").write_text(render(rep, cv))
    print("Written: reports/signal_diagnosis.md and signal_diagnosis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
