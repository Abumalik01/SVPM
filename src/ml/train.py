"""Phase D: cross-validation of the five approved models on DEVELOPMENT data only.

* Uses project-grouped folds from split_assignment.csv; the sealed test partition is never loaded.
* Primary and secondary datasets are run separately with the identical project assignment.
* Also runs the approved diagnostic baselines: majority dummy and label shuffle (training labels only).
* Applies the fixed U9 selection rule to the development results.

Run:  python -m src.ml.train            (add --shuffles 2 for a quick trial)
Output: reports/cv_results.json and reports/cv_summary.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time

import numpy as np

from src.ml import baselines, evaluate, features, pipeline, selection

SEED = 42


def _fold_metrics(model, Xva, yva):
    return evaluate.compute_metrics(yva, model.predict(Xva), pipeline.score(model, Xva))


def _agg(values):
    v = np.asarray(values, dtype=float)
    return {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v, ddof=1)),
            "se": float(np.nanstd(v, ddof=1) / np.sqrt(len(v)))}


def summarise_folds(folds: list[dict]) -> dict:
    return {m: _agg([f[m] for f in folds]) for m in ("precision", "recall", "f1", "roc_auc", "average_precision", "recall_top10pct")}


def _say(verbose, msg):
    if verbose:
        print(msg, flush=True)


def run_cv(dev, model_names=pipeline.MODEL_NAMES, n_shuffles=baselines.N_SHUFFLES, seed=SEED, verbose=False,
           feature_names=None, log_transform=False) -> dict:
    names = list(feature_names or features.FEATURE_NAMES)
    features.check_features(dev, names)
    X_all, y_all = dev[names].to_numpy(float), dev.label.to_numpy(int)
    fold_ids = sorted(dev.fold.unique())
    out = {"folds": [], "models": {}, "dummy": {}, "label_shuffle": {}}
    splits_ = []
    for k in fold_ids:
        va = (dev.fold == k).to_numpy()
        splits_.append((k, ~va, va))
        out["folds"].append({"fold": int(k), "train_rows": int((~va).sum()), "validation_rows": int(va.sum()),
                             "train_projects": int(dev[~va].repo_url.nunique()),
                             "validation_projects": int(dev[va].repo_url.nunique())})
    prev = [float((y_all[va] == 1).mean()) for _, _, va in splits_]
    out["reference"] = {"mean_prevalence": float(np.mean(prev)),
                        "mean_always_positive_f1": float(np.mean([2 * p / (1 + p) for p in prev])),
                        "average_precision_of_random_scorer": float(np.mean(prev)),
                        "note": "reference values, not models"}
    # approved candidate models
    for name in model_names:
        fm, t0 = [], time.time()
        for k, tr, va in splits_:
            m = pipeline.make_model(name, y_all[tr], seed, log_transform).fit(X_all[tr], y_all[tr])
            fm.append(_fold_metrics(m, X_all[va], y_all[va]))
            _say(verbose, f"  {name}: fold {k} done ({time.time() - t0:.0f}s so far)")
        out["models"][name] = {"folds": fm, "summary": summarise_folds(fm), "seconds": round(time.time() - t0, 1)}
    # diagnostic baseline 1: majority dummy
    dm = []
    for k, tr, va in splits_:
        d = baselines.majority_dummy().fit(X_all[tr], y_all[tr])
        dm.append(evaluate.compute_metrics(y_all[va], d.predict(X_all[va]), d.predict_proba(X_all[va])[:, 1]))
    out["dummy"] = {"folds": dm, "summary": summarise_folds(dm)}
    # diagnostic baseline 2: label shuffle (training labels only; validation labels untouched)
    for name in model_names:
        f1s, aucs = [], []
        _say(verbose, f"  label shuffle: {name} ...")
        for k, tr, va in splits_:
            for s in baselines.shuffle_seeds(int(k), seed, n_shuffles):
                y_sh = baselines.shuffle_labels(y_all[tr], s)
                m = pipeline.make_model(name, y_sh, seed, log_transform).fit(X_all[tr], y_sh)
                r = _fold_metrics(m, X_all[va], y_all[va])
                f1s.append(r["f1"]); aucs.append(r["roc_auc"])
        out["label_shuffle"][name] = {"n_fits": len(f1s), "f1": _agg(f1s), "roc_auc": _agg(aucs)}
    # approved selection rule U9 (development results only)
    cv_f1 = {n: [f["f1"] for f in out["models"][n]["folds"]] for n in model_names}
    cv_rec = {n: [f["recall"] for f in out["models"][n]["folds"]] for n in model_names}
    out["selection"] = selection.select_model(cv_f1, cv_rec) if len(model_names) == 5 else None
    return out


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def render(rep: dict) -> str:
    L = ["# Cross-validation results (development data only)", "",
         rep.get("label_note", "Label 1 = pre-fix method; label 0 = post-fix version of the same changed method. "
                 "These are not independently verified vulnerable and safe code."),
         f"Seed {rep['seed']}; shuffles per fold and model: {rep['n_shuffles']}; sealed test data used: **no**.", ""]
    for ds in ("primary", "secondary"):
        if ds not in rep:
            continue
        r = rep[ds]
        L += [f"## {rep.get('titles', {}).get(ds, ds.capitalize() + ' dataset')}", "",
              "| Fold | Train rows | Val rows | Train projects | Val projects |", "|---|---|---|---|---|"]
        L += [f"| {f['fold']} | {f['train_rows']:,} | {f['validation_rows']:,} | {f['train_projects']} | {f['validation_projects']} |" for f in r["folds"]]
        ref = r.get("reference")
        if ref:
            L += ["", f"Reference values (not models): mean prevalence of label 1 = {ref['mean_prevalence']:.3f}; "
                      f"F1 of labelling everything label 1 = {ref['mean_always_positive_f1']:.3f}; "
                      f"average precision of a random scorer = {ref['average_precision_of_random_scorer']:.3f}."]
        L += ["", "| Model | F1 (label 1) mean ± SE | Recall mean | Precision mean | ROC-AUC mean | Avg precision (suppl.) | Recall in top 10% (suppl.) | Predicted-positive rate |",
              "|---|---|---|---|---|---|---|---|"]
        for n, v in r["models"].items():
            s = v["summary"]
            ppr = np.mean([(f["confusion_matrix"][0][1] + f["confusion_matrix"][1][1]) / f["n"] for f in v["folds"]])
            L.append(f"| {n} | {s['f1']['mean']:.3f} ± {s['f1']['se']:.3f} | {s['recall']['mean']:.3f} | {s['precision']['mean']:.3f} | "
                     f"{s['roc_auc']['mean']:.3f} | {s['average_precision']['mean']:.3f} | {s['recall_top10pct']['mean']:.2f} | {ppr:.2f} |")
        d = r["dummy"]["summary"]
        L.append(f"| *Majority dummy (diagnostic)* | {d['f1']['mean']:.3f} | {d['recall']['mean']:.3f} | {d['precision']['mean']:.3f} | "
                 f"{d['roc_auc']['mean']:.3f} | {d['average_precision']['mean']:.3f} | {d['recall_top10pct']['mean']:.2f} | 0.00 |")
        L += ["", "Label-shuffle diagnostic (trained on shuffled labels, evaluated on real validation labels):", "",
              "| Model | Shuffled F1 mean | Shuffled ROC-AUC mean ± SD | Fits |", "|---|---|---|---|"]
        for n, v in r["label_shuffle"].items():
            L.append(f"| {n} | {v['f1']['mean']:.3f} | {v['roc_auc']['mean']:.3f} ± {v['roc_auc']['sd']:.3f} | {v['n_fits']} |")
        sel = r["selection"]
        L += ["", f"**U9 selection:** {sel['selected']} ({sel['decided_by']}).",
              f"Best by F1: {sel['best_by_f1']}, SE {sel['se_of_best']:.4f}, one-SE threshold {sel['one_se_threshold']:.3f}; "
              f"within one SE: {', '.join(sel['within_one_se'])}.", ""]
    L += ["## Reproducibility", "", json.dumps(rep["environment"], indent=2), ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffles", type=int, default=None, help="label shuffles per fold and model")
    ap.add_argument("--universe", action="store_true", help="use the function dataset (target v2)")
    ap.add_argument("--features", choices=sorted(features.FEATURE_SETS), default="v1_five", help="feature set")
    ap.add_argument("--log-transform", action="store_true", help="log1p before scaling for Logistic Regression and SVM")
    a = ap.parse_args()
    import sklearn, pandas, xgboost
    from src.data import settings as S
    proc = S.ROOT / "data" / "processed"
    t0 = time.time()
    if a.universe:
        csv, asg, tag = proc / "functions_capped.csv", proc / "split_assignment_v2.csv", "_v2"
        if a.features != "v1_five" or a.log_transform:
            tag += "_" + a.features.replace("v2_", "") + ("_log" if a.log_transform else "")
        n_shuf = a.shuffles if a.shuffles is not None else 3
        datasets = [("primary", features.load_development_v2(csv, asg, features.FEATURE_SETS[a.features]))]
        extra = {"titles": {"primary": "Function dataset (target v2, development data only)"},
                 "label_note": "Label 1 = function changed by the CVE-fixing commit; label 0 = other function in the same "
                               "pre-fix file. These are not independently verified vulnerable and safe functions."}
    else:
        csv, asg, tag = proc / "methods_c_cpp.csv", proc / "split_assignment.csv", ""
        n_shuf = a.shuffles if a.shuffles is not None else baselines.N_SHUFFLES
        p, s = features.load_development(csv, asg)
        datasets, extra = [("primary", p), ("secondary", s)], {}
    rep = {"seed": SEED, "n_shuffles": n_shuf, **extra}
    for name, d in datasets:
        print(f"Running {name} dataset ({len(d):,} development rows, {int((d.label == 1).sum()):,} label 1) ...", flush=True)
        rep[name] = run_cv(d, n_shuffles=n_shuf, verbose=True, feature_names=features.FEATURE_SETS[a.features],
                           log_transform=a.log_transform)
    rep["environment"] = {"features": features.FEATURE_SETS[a.features], "log_transform": a.log_transform,
                          "python": platform.python_version(), "scikit_learn": sklearn.__version__,
                          "xgboost": xgboost.__version__, "pandas": pandas.__version__, "numpy": np.__version__,
                          "hyperparameters": pipeline.HYPERPARAMETERS, "csv_sha256": _sha(csv),
                          "split_assignment_sha256": _sha(asg), "seconds": round(time.time() - t0, 1)}
    S.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (S.REPORT_DIR / f"cv_results{tag}.json").write_text(json.dumps(rep, indent=2, default=str))
    (S.REPORT_DIR / f"cv_summary{tag}.md").write_text(render(rep))
    for name, _ in datasets:
        print(name, "selected:", rep[name]["selection"]["selected"])
    print(f"Written: reports/cv_summary{tag}.md and reports/cv_results{tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
