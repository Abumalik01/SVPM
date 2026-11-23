"""Target v2, step 2: compare cap options for the function universe (NO model training, no files changed).

Why: one project holds 42.1% of all functions and 30.4% of the label-1 functions, so grouped
cross-validation would be impossible without caps. This script only REPORTS the effect of caps.

Selection criteria, fixed BEFORE seeing the grid (advisory, my engineering assumptions):
  1. the biggest project holds at most 10% of all functions AND at most 10% of label-1 functions;
  2. among options that satisfy (1), keep the MOST label-1 functions; ties: keep more functions.
No artificial balancing: the natural minority share of label 1 is kept (capping label 0 per project is
only to limit dominant projects).

Caps (seeded random choice, seed 42):
  positives per commit  -> limits large cleanup commits;  positives per project;  negatives per project.

Run:  python -m src.data.universe_caps
Output: reports/universe_caps.json and reports/universe_caps.md
"""
from __future__ import annotations

import itertools
import json
import math

import pandas as pd

from src.data.build_universe import summarise

SEED = 42
POS_COMMIT_CAPS = [None, 50]
POS_PROJECT_CAPS = [None, 300, 150]
NEG_PROJECT_CAPS = [None, 2000, 1000, 500]
MAX_TOP_SHARE = 0.10


def cap_group(df: pd.DataFrame, by: str, cap, seed: int = SEED) -> pd.DataFrame:
    if cap is None:
        return df
    return df.sample(frac=1.0, random_state=seed).groupby(by, sort=False).head(cap)


def apply_caps(df: pd.DataFrame, pos_commit, pos_project, neg_project, seed: int = SEED) -> pd.DataFrame:
    pos = cap_group(cap_group(df[df.label == 1], "commit_hash", pos_commit, seed), "repo_url", pos_project, seed)
    neg = cap_group(df[df.label == 0], "repo_url", neg_project, seed)
    return pd.concat([pos, neg]).sort_values("function_id").copy()


def evaluate_option(df: pd.DataFrame, pos_commit, pos_project, neg_project) -> dict:
    d = apply_caps(df, pos_commit, pos_project, neg_project)
    s = summarise(d)
    worst = max(s["top_project_share"] or 0, s["top_project_share_of_positives"] or 0)
    s.update(pos_commit_cap=pos_commit, pos_project_cap=pos_project, neg_project_cap=neg_project,
             max_k_for_grouped_folds=int(math.floor(1 / worst)) if worst else None,
             meets_criteria=bool((s["top_project_share"] or 1) <= MAX_TOP_SHARE
                                 and (s["top_project_share_of_positives"] or 1) <= MAX_TOP_SHARE))
    return s


def choose(options: list[dict]) -> dict | None:
    ok = [o for o in options if o["meets_criteria"]]
    if not ok:
        return None
    return max(ok, key=lambda o: (o["label_1_changed"], o["functions"]))


def grid(df: pd.DataFrame) -> dict:
    opts = [evaluate_option(df, pc, pp, npj) for pc, pp, npj in
            itertools.product(POS_COMMIT_CAPS, POS_PROJECT_CAPS, NEG_PROJECT_CAPS)]
    best = choose(opts)
    return {"criteria": {"max_top_project_share": MAX_TOP_SHARE, "rule": "most label-1 functions among options that pass"},
            "seed": SEED, "options": opts, "recommended": best,
            "note_if_none_pass": None if best else "No option meets the criteria; widen the grid or relax the criteria."}


def render(rep: dict) -> str:
    pct = lambda x: "n/a" if x is None else f"{x:.1%}"
    rec = rep["recommended"]
    L = ["# Cap options for the function universe (report only)", "",
         f"Criteria fixed in advance: top project at most {rep['criteria']['max_top_project_share']:.0%} of all functions and of "
         "label-1 functions; then keep the most label-1 functions. Seed 42.", ""]
    L.append("**Recommended by the rule:** " + ("none passes" if rec is None else
             f"positives per commit {rec['pos_commit_cap']}, per project {rec['pos_project_cap']}; negatives per project {rec['neg_project_cap']}"))
    L += ["", "| Pos/commit | Pos/project | Neg/project | Functions | Label 1 | Positive share | Projects | Top share (all) | Top share (label 1) | Max folds | Passes |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for o in rep["options"]:
        L.append(f"| {o['pos_commit_cap']} | {o['pos_project_cap']} | {o['neg_project_cap']} | {o['functions']:,} | {o['label_1_changed']:,} | "
                 f"{pct(o['positive_share'])} | {o['projects']} | {pct(o['top_project_share'])} | {pct(o['top_project_share_of_positives'])} | "
                 f"{o['max_k_for_grouped_folds']} | {'yes' if o['meets_criteria'] else 'no'} |")
    return "\n".join(L)


def main() -> int:
    from src.data import settings as S
    df = pd.read_csv(S.INTERIM_DIR / "function_universe.csv")
    rep = grid(df)
    S.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (S.REPORT_DIR / "universe_caps.json").write_text(json.dumps(rep, indent=2, default=str))
    (S.REPORT_DIR / "universe_caps.md").write_text(render(rep))
    print("Written: reports/universe_caps.md and universe_caps.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
