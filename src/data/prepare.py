"""Phase D, step 1: build the processed C/C++ subset (NO model training here).

Applies the approved cleaning rules by reusing functions from subset_audit.py:
  language C/C++  ->  remove code texts with both labels  ->  remove code texts
  shared by several projects  ->  keep one copy of repeats inside a project
  ->  pair-aware cap of PROJECT_CAP rows per label per project.

Pair-aware means: a before/after pair of the same method (same file and
signature) is kept or dropped as one unit, so pairs are not split by the cap.

Run:  python -m src.data.prepare
Output: data/processed/methods_c_cpp.csv   (no source code, only metrics/ids)
        reports/prepare_report.json and reports/prepare_summary.md
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import subset_audit as sa

PROJECT_CAP = 500                      # approved decision D9
SEED = 42
FEATURES = ["nloc", "complexity", "token_count", "top_nesting_level", "n_params"]
KEEP_COLUMNS = ["method_change_id", "file_change_id", "repo_url", "commit_hash",
                "signature", "language", "label", *FEATURES, "code_hash",
                "unit", "pair_complete"]


def add_units(df: pd.DataFrame) -> pd.DataFrame:
    """unit = one method identity (file + signature); a pair shares a unit."""
    df = df.copy()
    df["unit"] = df.groupby(["file_change_id", "signature"]).ngroup()
    return df


def pair_aware_cap(df: pd.DataFrame, cap: int = PROJECT_CAP, seed: int = SEED) -> pd.DataFrame:
    """Keep at most `cap` rows per label in each project, choosing whole units
    in a seeded random order. A unit is skipped if keeping it would push either
    label above the cap."""
    df = add_units(df)
    rng = np.random.default_rng(seed)
    kept = []
    for _, g in df.groupby("repo_url", sort=True):
        u = g.groupby("unit")["label"].agg(n1=lambda s: int((s == 1).sum()),
                                           n0=lambda s: int((s == 0).sum()))
        order = rng.permutation(len(u))
        c1 = c0 = 0
        for idx in order:
            unit_id, row = u.index[idx], u.iloc[idx]
            if c1 + row.n1 <= cap and c0 + row.n0 <= cap:
                kept.append(unit_id)
                c1 += row.n1
                c0 += row.n0
    return df[df.unit.isin(kept)].copy()


def flag_complete_pairs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    both = df.groupby("unit")["label"].transform("nunique") == 2
    df["pair_complete"] = both
    return df


def build_subset(df_all: pd.DataFrame, cap: int = PROJECT_CAP, seed: int = SEED):
    steps, clean = sa.attrition(df_all)
    capped = flag_complete_pairs(pair_aware_cap(clean, cap, seed))
    return steps, capped


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def render(rep: dict) -> str:
    pct = lambda x: "n/a" if x is None else f"{x:.1%}"
    L = ["# Processed subset report", "",
         f"Cap: {rep['project_cap']} rows per label per project (pair-aware); seed {rep['seed']}", "",
         "| Step | Rows | Pre-fix | Post-fix | Projects |", "|---|---|---|---|---|"]
    for name, s in rep["attrition"]:
        L.append(f"| {name} | {s['rows']:,} | {s['pre_fix_1']:,} | {s['post_fix_0']:,} | {s['projects']:,} |")
    f = rep["final"]
    L += ["", "## Final primary dataset", "",
          f"- rows: {f['rows']:,} (pre-fix {f['pre_fix_1']:,}, post-fix {f['post_fix_0']:,})",
          f"- projects: {f['projects']:,}; commits: {f['commits']:,}",
          f"- top project share: {pct(f['top_project_share'])}; max folds: {f['max_k_for_grouped_folds']}",
          f"- complete before/after pairs: {f['complete_before_after_pairs']:,}",
          "", "## Secondary dataset (complete pairs only)", "",
          f"- rows: {rep['secondary']['rows']:,} (pre-fix {rep['secondary']['pre_fix_1']:,}, "
          f"post-fix {rep['secondary']['post_fix_0']:,}); projects: {rep['secondary']['projects']:,}",
          "", "## Languages in the final dataset", ""]
    L += [f"- {k}: {v:,}" for k, v in rep["language_rows"].items()]
    L += ["", f"CSV sha256: `{rep['csv_sha256']}`", ""]
    return "\n".join(L)


def run() -> dict:
    from src.data import audit, settings as S            # needs the real database
    out_csv = S.ROOT / "data" / "processed" / "methods_c_cpp.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    con = audit.connect_ro(S.DB_PATH)
    try:
        df = audit.load_method_frame(con).dropna(subset=["label"])
    finally:
        con.close()
    steps, final = build_subset(df)
    final[KEEP_COLUMNS].to_csv(out_csv, index=False)
    sec = final[final.pair_complete]
    rep = {"project_cap": PROJECT_CAP, "seed": SEED, "attrition": steps,
           "final": sa.summarise(final), "secondary": sa.summarise(sec),
           "language_rows": {k: int(v) for k, v in final.language.value_counts().items()},
           "csv_path": str(out_csv), "csv_sha256": sha256_file(out_csv)}
    S.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (S.REPORT_DIR / "prepare_report.json").write_text(json.dumps(rep, indent=2, default=str))
    (S.REPORT_DIR / "prepare_summary.md").write_text(render(rep))
    return rep


if __name__ == "__main__":
    run()
    print("Written: data/processed/methods_c_cpp.csv, reports/prepare_summary.md")
