"""THE single evaluation on the sealed test set (run once, after every choice is frozen).

Frozen before this run (see docs/evaluation_protocol.md, training_config_v3.md):
  * features, log transform and settings = those of the cross-validation run in reports/cv_results_v2_four_log.json;
  * PRIMARY model = the U9 winner in that file. Whatever the test shows, the prototype model is the primary model;
  * SECONDARY model = Logistic Regression, declared in advance for transparency only (no switching afterwards);
  * references: majority dummy, labelling everything label 1, random scorer, and token_count alone (direction chosen
    on development data).
Models are fitted on ALL development data. Confidence intervals resample WHOLE test projects (1,000 resamples, seed 42).

Safeguards: the command needs --confirm-final; a lock file is written when the run starts and a second run is refused
(--allow-rerun exists only for a documented repeat). If a run fails before producing results, the lock is removed.

Run:  python -m src.ml.final_evaluation --confirm-final [--promote]
Output: reports/final_evaluation_v2.json and .md; the primary model is saved to the registry as a candidate
        (sealed_test_evaluated = true) and promoted with --promote.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.ml import baselines, evaluate, features, final_fit, pipeline, registry, splits

SECONDARY_MODEL = "LogisticRegression"          # declared in advance
N_BOOT = 1000
SEED = 42


def _table(test: pd.DataFrame, y_pred, y_score) -> pd.DataFrame:
    return pd.DataFrame({"repo_url": test.repo_url.to_numpy(), "y_true": test.label.to_numpy(int),
                         "y_pred": np.asarray(y_pred, int), "y_score": np.asarray(y_score, float)})


def _result(table: pd.DataFrame, n_boot: int, seed: int) -> dict:
    r = evaluate.bootstrap_ci_by_project(table, n_boot=n_boot, seed=seed)
    return {"point": r["point_estimate"], "ci95": r["ci95"], "n_boot": r["n_boot"],
            "n_skipped_single_class": r["n_skipped_single_class"], "n_projects_resampled": r["n_projects_resampled"]}


def per_language(test: pd.DataFrame, y_score) -> dict:
    out = {}
    for lang, idx in test.reset_index(drop=True).groupby("language").groups.items():
        y, s = test.label.to_numpy(int)[list(idx)], np.asarray(y_score)[list(idx)]
        out[str(lang)] = {"functions": int(len(y)), "label_1": int(y.sum()),
                          "roc_auc": float(roc_auc_score(y, s)) if len(set(y)) == 2 else None,
                          "recall_top10pct": evaluate.recall_at_top(y, s)}
    return out


def evaluate_models(dev: pd.DataFrame, test: pd.DataFrame, primary: str, names, log_transform: bool,
                    n_boot: int = N_BOOT, seed: int = SEED):
    """Returns (report dict, fitted primary model, its metadata)."""
    Xd, yd = dev[names].to_numpy(float), dev.label.to_numpy(int)
    Xt, yt = test[names].to_numpy(float), test.label.to_numpy(int)
    rep = {"test": {"functions": int(len(test)), "label_1": int(yt.sum()), "projects": int(test.repo_url.nunique()),
                    "prevalence": float(yt.mean())}, "models": {}, "references": {}}
    fitted = {}
    for role, name in (("primary", primary), ("secondary", SECONDARY_MODEL)):
        if role == "secondary" and name == primary:
            continue
        m = pipeline.make_model(name, yd, seed, log_transform).fit(Xd, yd)
        s = pipeline.score(m, Xt)
        rep["models"][role] = {"name": name, **_result(_table(test, m.predict(Xt), s), n_boot, seed),
                               "per_language": per_language(test, s)}
        fitted[role] = m
    # references
    dummy = baselines.majority_dummy().fit(Xd, yd)
    rep["references"]["majority_dummy"] = _result(_table(test, dummy.predict(Xt), dummy.predict_proba(Xt)[:, 1]), n_boot, seed)
    p = float(yt.mean())
    rep["references"]["label_everything_1"] = {"f1": 2 * p / (1 + p), "precision": p, "recall": 1.0}
    rep["references"]["random_scorer_average_precision"] = p
    d = 1 if roc_auc_score(yd, dev.token_count) >= 0.5 else -1          # direction fixed on development data
    ss = d * test.token_count.to_numpy(float)
    rep["references"]["token_count_alone"] = {"direction": d, **_result(_table(test, np.zeros(len(test)), ss), n_boot, seed)}
    # the very model that was evaluated is the one saved for the prototype (no refit)
    model, meta = final_fit.build_bundle(dev, primary, names, log_transform, model=fitted["primary"])
    return rep, model, meta


def render(rep: dict) -> str:
    t = rep["test"]
    fmt = lambda r, k: f"{r['point'][k]:.3f} ({r['ci95'][k][0]:.3f} to {r['ci95'][k][1]:.3f})" if k in r["ci95"] else f"{r['point'][k]:.3f}"
    L = ["# Final evaluation on the sealed test set (single run)", "",
         "Label 1 = function changed by the CVE-fixing commit; label 0 = other function in the same pre-fix file. "
         "These are not independently verified vulnerable and safe functions.", "",
         f"Test set: {t['functions']:,} functions, {t['label_1']:,} label 1 ({t['prevalence']:.1%}), {t['projects']} projects. "
         f"95% intervals resample whole projects ({rep['n_boot']} resamples, seed {rep['seed']}).",
         f"Primary model (fixed by the U9 rule before this run): **{rep['models']['primary']['name']}**. "
         f"Secondary (declared in advance, reported for transparency only): {rep['models'].get('secondary', {}).get('name', 'none')}.", "",
         "| Model | ROC-AUC | Avg precision | Recall in top 10% | Precision | Recall | F1 |", "|---|---|---|---|---|---|---|"]
    for role, r in rep["models"].items():
        L.append(f"| {r['name']} ({role}) | {fmt(r,'roc_auc')} | {fmt(r,'average_precision')} | {fmt(r,'recall_top10pct')} | "
                 f"{fmt(r,'precision')} | {fmt(r,'recall')} | {fmt(r,'f1')} |")
    ref = rep["references"]
    ta = ref["token_count_alone"]
    L.append(f"| token_count alone (reference) | {fmt(ta,'roc_auc')} | {fmt(ta,'average_precision')} | {fmt(ta,'recall_top10pct')} | n/a | n/a | n/a |")
    L += ["", f"Reference values: random-scorer average precision {ref['random_scorer_average_precision']:.3f}; labelling everything label 1 gives "
              f"F1 {ref['label_everything_1']['f1']:.3f} (precision {ref['label_everything_1']['precision']:.3f}); majority dummy F1 "
              f"{ref['majority_dummy']['point']['f1']:.3f}.", "", "## Primary model by language (point estimates)", "",
          "| Language | Functions | Label 1 | ROC-AUC | Recall in top 10% |", "|---|---|---|---|---|"]
    for lang, v in rep["models"]["primary"]["per_language"].items():
        auc = "n/a" if v["roc_auc"] is None else f"{v['roc_auc']:.3f}"
        L.append(f"| {lang} | {v['functions']:,} | {v['label_1']:,} | {auc} | {v['recall_top10pct']:.2f} |")
    L += ["", "How to read: precision depends on the share of label 1 in this test set, which is higher than in real software "
              "because large projects were capped. Rankings (ROC-AUC, average precision, recall in the top 10%) are more transferable. "
              "Any signal is mainly function size (see the size baseline)."]
    return "\n".join(L)


def run(confirm: bool, promote: bool = False, allow_rerun: bool = False) -> dict:
    from src import paths
    if not confirm:
        raise SystemExit("The sealed test set is opened only for the final evaluation. Re-run with --confirm-final.")
    lock = paths.REPORT_DIR / "final_evaluation_v2.lock"
    if lock.exists() and not allow_rerun:
        raise SystemExit(f"{lock.name} exists: the sealed test set was already used. A repeat must be justified and disclosed "
                         "(use --allow-rerun and record it in the decision log).")
    paths.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lock.write_text(dt.datetime.now(dt.timezone.utc).isoformat())
    try:
        cv = json.loads((paths.REPORT_DIR / "cv_results_v2_four_log.json").read_text())
        primary = cv["primary"]["selection"]["selected"]
        env = cv["environment"]
        names, log_t = env["features"], env["log_transform"]
        df = pd.read_csv(paths.PROCESSED_DIR / "functions_capped.csv")
        assigned = splits.apply_assignment(df, pd.read_csv(paths.PROCESSED_DIR / "split_assignment_v2.csv"))
        dev = features.development_only(assigned, names)
        test = splits.get_test(assigned, confirm_final_evaluation=True).reset_index(drop=True)
        features.check_features(test.assign(fold=0), names)
        rep, model, meta = evaluate_models(dev, test, primary, names, log_t)
        rep.update({"seed": SEED, "n_boot": N_BOOT, "features": names, "log_transform": log_t,
                    "primary_selected_by": cv["primary"]["selection"]["decided_by"],
                    "run_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
        meta.update({"note": "Final model: fitted on all development data and evaluated once on the sealed test set.",
                     "selection_trace": cv["primary"]["selection"]})
        pm = rep["models"]["primary"]
        meta["sealed_test_evaluated"] = True
        meta["test_results"] = {"roc_auc": pm["point"]["roc_auc"], "average_precision": pm["point"]["average_precision"],
                                "recall_top10pct": pm["point"]["recall_top10pct"], "ci95": pm["ci95"],
                                "report": "reports/final_evaluation_v2.md"}
        version = registry.save_candidate(model, meta, paths.MODELS_DIR)
        if promote:
            registry.promote(version, paths.MODELS_DIR)
        rep["saved_version"], rep["promoted"] = version, bool(promote)
    except BaseException:
        lock.unlink(missing_ok=True)       # a failed run produced no results, so the test set was not "used"
        raise
    (paths.REPORT_DIR / "final_evaluation_v2.json").write_text(json.dumps(rep, indent=2, default=str))
    (paths.REPORT_DIR / "final_evaluation_v2.md").write_text(render(rep))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-final", action="store_true")
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--allow-rerun", action="store_true")
    a = ap.parse_args()
    r = run(a.confirm_final, a.promote, a.allow_rerun)
    print("Final evaluation done. Saved model", r["saved_version"], "(active)" if r["promoted"] else "(candidate)")
    print("Written: reports/final_evaluation_v2.md and final_evaluation_v2.json")
