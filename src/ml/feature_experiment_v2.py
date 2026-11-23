"""Feature-set experiment (development data only; no sealed test data; nothing is saved as a model).

Question: can any feature set beat plain function size under the same project-grouped folds?

Feature sets (S0 and S1 are references, S2 to S5 are candidates):
  S0 token_count only                          (reference: pure size)
  S1 the five approved Lizard features         (reference: current model inputs)
  S2 four features: S1 without top_nesting_level
  S3 size and density: nloc, complexity, token_count, n_params, complexity per line, tokens per line
  S4 S3 + 13 code-pattern counts (lexical_features.py)
  S5 S4 + the same counts per 100 tokens
Models: Logistic Regression (log1p transform + scaling inside the pipeline, same settings as the approved model)
and XGBoost (approved settings). Metrics per fold: ROC-AUC, average precision, and recall in the top 10% of functions.

RULE FIXED BEFORE THE RUN: among candidates S2 to S5, take the best mean average precision under Logistic Regression;
every candidate within one standard error of that best (SE of the best set's five fold values) is eligible; choose the
eligible set with the FEWEST features (ties: higher mean AP). XGBoost results are shown to check non-linear gains only.

Run:  python -m src.ml.feature_experiment_v2
Output: reports/feature_experiment_v2.json and .md
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.ml import features, pipeline
from src.ml.lexical_features import LEX_NAMES

F = features.FEATURE_NAMES
SEED = 42
CANDIDATES = ["S2 four features", "S3 size and density", "S4 S3 + code-pattern counts", "S5 S4 + densities"]


def design(df: pd.DataFrame, name: str) -> pd.DataFrame:
    nloc = df.nloc.clip(lower=1)
    dens = pd.DataFrame({"complexity_per_line": df.complexity / nloc, "tokens_per_line": df.token_count / nloc})
    base = df[["nloc", "complexity", "token_count", "n_params"]]
    if name == "S0 token_count only":
        return df[["token_count"]]
    if name == "S1 approved five":
        return df[F]
    if name == "S2 four features":
        return df[[f for f in F if f != "top_nesting_level"]]
    s3 = pd.concat([base, dens], axis=1)
    if name == "S3 size and density":
        return s3
    lex = df[LEX_NAMES]
    if name == "S4 S3 + code-pattern counts":
        return pd.concat([s3, lex], axis=1)
    if name == "S5 S4 + densities":
        per100 = lex.div(df.token_count.clip(lower=1), axis=0).mul(100).add_suffix("_per100tok")
        return pd.concat([s3, lex, per100], axis=1)
    raise ValueError(name)


def _lr(seed=SEED):
    return make_pipeline(FunctionTransformer(np.log1p, validate=False), StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=seed))


def recall_at_top(y, s, frac=0.10) -> float:
    k = max(1, math.ceil(frac * len(y)))
    idx = np.argsort(-np.asarray(s), kind="stable")[:k]
    return float(np.asarray(y)[idx].sum() / max(np.asarray(y).sum(), 1))


def _agg(v):
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v))), "folds": [float(x) for x in v]}


def run_experiment(dev: pd.DataFrame, set_names: list[str], seed: int = SEED) -> dict:
    features.check_features(dev)
    y_all = dev.label.to_numpy(int)
    out = {}
    for sn in set_names:
        X = design(dev, sn).to_numpy(float)
        res = {"n_features": int(X.shape[1])}
        for mname in ("LogisticRegression", "XGBoost"):
            auc, ap, rec = [], [], []
            for k in sorted(dev.fold.unique()):
                va = (dev.fold == k).to_numpy()
                m = (_lr(seed) if mname == "LogisticRegression" else pipeline.make_model("XGBoost", y_all[~va], seed))
                m.fit(X[~va], y_all[~va])
                s = pipeline.score(m, X[va])
                auc.append(roc_auc_score(y_all[va], s)); ap.append(average_precision_score(y_all[va], s))
                rec.append(recall_at_top(y_all[va], s))
            res[mname] = {"auc": _agg(auc), "ap": _agg(ap), "recall_top10pct": _agg(rec)}
        out[sn] = res
    return out


def choose_feature_set(results: dict) -> dict | None:
    cands = {n: r for n, r in results.items() if n in CANDIDATES}
    if not cands:
        return None
    best = max(cands, key=lambda n: cands[n]["LogisticRegression"]["ap"]["mean"])
    thr = cands[best]["LogisticRegression"]["ap"]["mean"] - cands[best]["LogisticRegression"]["ap"]["se"]
    eligible = [n for n, r in cands.items() if r["LogisticRegression"]["ap"]["mean"] >= thr]
    chosen = min(eligible, key=lambda n: (cands[n]["n_features"], -cands[n]["LogisticRegression"]["ap"]["mean"]))
    return {"best_by_mean_ap": best, "one_se_threshold": thr, "eligible": eligible, "chosen": chosen}


def render(rep: dict) -> str:
    L = ["# Feature-set experiment (development data only; sealed test not used)", "",
         f"Random-scorer average precision = prevalence = {rep['mean_prevalence']:.3f}. Mean of 5 project-grouped folds ± SE.", "",
         "| Feature set | Features | LR ROC-AUC | LR avg precision | LR recall in top 10% | XGB ROC-AUC | XGB avg precision | XGB recall in top 10% |",
         "|---|---|---|---|---|---|---|---|"]
    for n, r in rep["results"].items():
        lr, xg = r["LogisticRegression"], r["XGBoost"]
        L.append(f"| {n} | {r['n_features']} | {lr['auc']['mean']:.3f} ± {lr['auc']['se']:.3f} | {lr['ap']['mean']:.3f} ± {lr['ap']['se']:.3f} | "
                 f"{lr['recall_top10pct']['mean']:.2f} | {xg['auc']['mean']:.3f} ± {xg['auc']['se']:.3f} | {xg['ap']['mean']:.3f} ± {xg['ap']['se']:.3f} | "
                 f"{xg['recall_top10pct']['mean']:.2f} |")
    c = rep["choice"]
    L += ["", "**Rule (fixed in advance):** " + ("no candidate sets were run." if c is None else
          f"best by mean AP: {c['best_by_mean_ap']}; eligible within one SE: {', '.join(c['eligible'])}; fewest features among them: **{c['chosen']}**."),
          "", "S0 and S1 are references, not candidates. 'Recall in top 10%' = share of label-1 functions found among the 10% "
          "highest-scored functions of a fold (a review-prioritisation figure)."]
    return "\n".join(L)


def main() -> int:
    from src import paths
    p = paths.PROCESSED_DIR
    lex = p / "functions_capped_lex.csv"
    src = lex if lex.exists() else p / "functions_capped.csv"
    dev = features.development_only(__import__("src.ml.splits", fromlist=["x"]).apply_assignment(
        pd.read_csv(src), pd.read_csv(p / "split_assignment_v2.csv")))
    names = ["S0 token_count only", "S1 approved five", *CANDIDATES[:2]]
    if all(c in dev.columns for c in LEX_NAMES):
        names += CANDIDATES[2:]
    else:
        print("Code-pattern columns not found; run python -m src.data.add_lexical first to include S4 and S5.")
    res = run_experiment(dev, names)
    rep = {"mean_prevalence": float(np.mean([(g.label == 1).mean() for _, g in dev.groupby("fold")])),
           "results": res, "choice": choose_feature_set(res)}
    (paths.REPORT_DIR / "feature_experiment_v2.json").write_text(json.dumps(rep, indent=2))
    (paths.REPORT_DIR / "feature_experiment_v2.md").write_text(render(rep))
    print("Written: reports/feature_experiment_v2.md and feature_experiment_v2.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
