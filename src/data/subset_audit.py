"""Phase B, step 3: audit the APPROVED subset (C/C++, method level).

It reuses the existing audit.py (loader and connection helpers) and adds:
  1. A check of the 'unknown' language hypothesis (why those rows are all pre-fix).
  2. An attrition table: how many rows remain after each cleaning step.
  3. A sensitivity table for per-commit and per-project caps.
  4. Feasibility of project-grouped k-fold cross-validation.

Every number is computed from the database. Nothing is chosen automatically:
the caps are shown side by side so the student and supervisor can decide.

Run:  python -m src.data.subset_audit
Output: reports/subset_audit.json and reports/subset_audit.md
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

SUBSET_LANGUAGES = ("C", "C++")      # approved decision D6 (pending this check)
COMMIT_CAPS = [None, 100, 50, 25]    # max rows per (commit, label)
PROJECT_CAPS = [None, 500, 250]      # max rows per (project, label)
SEED = 42
K_FOLDS = 5


# --------------------------------------------------------------------------
# pure functions (tested without a database)
# --------------------------------------------------------------------------
def summarise(df: pd.DataFrame) -> dict:
    n1, n0 = int((df.label == 1).sum()), int((df.label == 0).sum())
    proj = df.groupby("repo_url").size().sort_values(ascending=False)
    top = float(proj.iloc[0] / proj.sum()) if proj.size else None
    pairs = df.groupby(["file_change_id", "signature"])["label"].nunique()
    return {
        "rows": int(len(df)), "pre_fix_1": n1, "post_fix_0": n0,
        "minority_share": float(min(n1, n0) / (n1 + n0)) if (n1 + n0) else None,
        "projects": int(proj.size), "commits": int(df.commit_hash.nunique()),
        "top_project_share": top,
        "top5_project_share": float(proj.head(5).sum() / proj.sum()) if proj.size else None,
        "complete_before_after_pairs": int((pairs == 2).sum()),
        # a project holding more than 1/k of the rows cannot be spread over k folds
        "max_k_for_grouped_folds": int(math.floor(1 / top)) if top else None,
    }


def filter_language(df: pd.DataFrame, languages=SUBSET_LANGUAGES) -> pd.DataFrame:
    return df[df.language.isin(languages)].copy()


def drop_label_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """Remove every row whose exact code text appears with both labels."""
    nlab = df.groupby("code_hash")["label"].transform("nunique")
    return df[nlab < 2].copy()


def drop_cross_project_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove code texts found in more than one project (vendored copies).
    They would cross fold boundaries even when splitting by project."""
    nproj = df.groupby("code_hash")["repo_url"].transform("nunique")
    return df[nproj < 2].copy()


def drop_within_project_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one copy of identical code text with the same label in a project."""
    return df.sort_values("method_change_id").drop_duplicates(
        subset=["repo_url", "code_hash", "label"], keep="first").copy()


def cap_rows(df: pd.DataFrame, by: str, cap: int | None, seed: int = SEED) -> pd.DataFrame:
    """Keep at most `cap` rows per (by, label), chosen with a fixed seed."""
    if cap is None:
        return df.copy()
    shuffled = df.sample(frac=1.0, random_state=seed)
    return shuffled.groupby([by, "label"], sort=False).head(cap).copy()


def attrition(df_all: pd.DataFrame, languages=SUBSET_LANGUAGES) -> tuple[list, pd.DataFrame]:
    steps = [("0 all method rows", summarise(df_all))]
    d = filter_language(df_all, languages)
    steps.append((f"1 language in {list(languages)}", summarise(d)))
    d = drop_label_conflicts(d)
    steps.append(("2 remove code texts carrying both labels", summarise(d)))
    d = drop_cross_project_duplicates(d)
    steps.append(("3 remove code texts shared by several projects", summarise(d)))
    d = drop_within_project_duplicates(d)
    steps.append(("4 keep one copy of repeats inside a project", summarise(d)))
    return steps, d


def cap_sensitivity(clean: pd.DataFrame) -> list[dict]:
    out = []
    for cc in COMMIT_CAPS:
        for pc in PROJECT_CAPS:
            d = cap_rows(cap_rows(clean, "commit_hash", cc), "repo_url", pc)
            row = {"commit_cap": cc, "project_cap": pc}
            row.update(summarise(d))
            out.append(row)
    return out


# --------------------------------------------------------------------------
# database part
# --------------------------------------------------------------------------
def unknown_language_check(con) -> dict:
    """Why are 'unknown' rows all pre-fix? Test the 'deleted file' idea."""
    cur = con.execute("""
        SELECT change_type, COUNT(*),
               SUM(CASE WHEN code_after IS NULL OR TRIM(code_after) = '' THEN 1 ELSE 0 END)
        FROM file_change WHERE programming_language = 'unknown'
        GROUP BY change_type""")
    rows = [{"change_type": r[0], "files": r[1], "files_with_empty_code_after": r[2]}
            for r in cur.fetchall()]
    return {"files_by_change_type": rows,
            "reading_guide": ("If almost all files are DELETE with empty code_after, the "
                              "'unknown' rows are methods from files removed by the fix, so "
                              "their language could not be guessed.")}


def render(rep: dict) -> str:
    def pct(x):
        return "n/a" if x is None else f"{x:.1%}"
    L = ["# Subset audit (C/C++, method level)", "", f"Seed: {SEED}", "",
         "## 1. 'unknown' language check", "", "| change_type | files | empty code_after |", "|---|---|---|"]
    L += [f"| {r['change_type']} | {r['files']} | {r['files_with_empty_code_after']} |"
          for r in rep["unknown_language"]["files_by_change_type"]]
    L += ["", rep["unknown_language"]["reading_guide"], "", "## 2. Attrition", "",
          "| Step | Rows | Pre-fix | Post-fix | Projects | Top share | Pairs |", "|---|---|---|---|---|---|---|"]
    for name, s in rep["attrition"]:
        L.append(f"| {name} | {s['rows']:,} | {s['pre_fix_1']:,} | {s['post_fix_0']:,} | "
                 f"{s['projects']:,} | {pct(s['top_project_share'])} | {s['complete_before_after_pairs']:,} |")
    L += ["", "## 3. Cap sensitivity (applied after step 4)", "",
          "| Commit cap | Project cap | Rows | Pre-fix | Post-fix | Projects | Top share | Max k |",
          "|---|---|---|---|---|---|---|---|"]
    for s in rep["cap_sensitivity"]:
        L.append(f"| {s['commit_cap']} | {s['project_cap']} | {s['rows']:,} | {s['pre_fix_1']:,} | "
                 f"{s['post_fix_0']:,} | {s['projects']:,} | {pct(s['top_project_share'])} | {s['max_k_for_grouped_folds']} |")
    L += ["", f"'Max k' = largest number of folds a grouped split can use before the biggest "
              f"project is more than one fold's share. Plan uses k={K_FOLDS}.", ""]
    return "\n".join(L)


def run(out_dir: Path | None = None) -> dict:
    from src.data import audit, settings as S   # imported here so tests need no database
    out_dir = Path(out_dir or S.REPORT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    con = audit.connect_ro(S.DB_PATH)
    try:
        df = audit.load_method_frame(con).dropna(subset=["label"])
        unknown = unknown_language_check(con)
    finally:
        con.close()
    steps, clean = attrition(df)
    rep = {"subset_languages": list(SUBSET_LANGUAGES), "seed": SEED,
           "unknown_language": unknown, "attrition": steps,
           "cap_sensitivity": cap_sensitivity(clean)}
    (out_dir / "subset_audit.json").write_text(json.dumps(rep, indent=2, default=str))
    (out_dir / "subset_audit.md").write_text(render(rep))
    return rep


if __name__ == "__main__":
    run()
    print("Subset audit written to reports/subset_audit.md and subset_audit.json")
