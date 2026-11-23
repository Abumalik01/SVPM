"""Phase D: project-grouped partitioning (approved decision U8). NO training here.

Structure: about 80% development / 20% sealed test, grouped by project, then
5-fold cross-validation grouped by project inside the development part.

Rules enforced and verified:
  * A project is in exactly one partition (and one fold if development).
  * Projects that share a commit hash (forks) are merged into one group first.
  * Before/after pairs (same file + signature) live inside one project, so they
    cannot cross partitions.
  * The SAME project assignment is applied to the primary and secondary datasets.
  * The test part is sealed: get_test() refuses unless explicitly unsealed.

Run:  python -m src.ml.splits
Output: data/processed/split_assignment.csv, reports/split_report.json, split_summary.md
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
N_FOLDS = 5
TEST_FRACTION_FOLDS = 5          # one of 5 outer folds = about 20% test
MIN_PER_LABEL_PER_FOLD = 30      # sanity floor reported by verify(), not a standard


# --------------------------------------------------------------------------
def make_groups(df: pd.DataFrame) -> pd.Series:
    """Union-find: projects sharing any commit hash get one group id."""
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for repo, commit in df[["repo_url", "commit_hash"]].drop_duplicates().itertuples(index=False):
        a, b = find(("r", repo)), find(("c", commit))
        if a != b:
            parent[a] = b
    roots = df.repo_url.map(lambda r: find(("r", r)))
    return pd.Series(pd.factorize(roots)[0], index=df.index, name="group_id")


def build_assignment(df: pd.DataFrame, seed: int = SEED, n_folds: int = N_FOLDS) -> pd.DataFrame:
    """Return one row per project: repo_url, group_id, partition, fold.
    Built from the PRIMARY dataset only; the secondary reuses it."""
    d = df.copy()
    d["group_id"] = make_groups(d)
    y, groups = d.label.values, d.group_id.values
    outer = StratifiedGroupKFold(n_splits=TEST_FRACTION_FOLDS, shuffle=True, random_state=seed)
    dev_idx, test_idx = next(outer.split(np.zeros(len(d)), y, groups))
    part = pd.Series("dev", index=d.index)
    part.iloc[test_idx] = "test"
    fold = pd.Series(-1, index=d.index)
    dev = d.iloc[dev_idx]
    inner = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, val_pos) in enumerate(inner.split(np.zeros(len(dev)), dev.label.values, dev.group_id.values)):
        fold.loc[dev.index[val_pos]] = k
    d["partition"], d["fold"] = part, fold
    out = d.groupby("repo_url").agg(group_id=("group_id", "first"), partition=("partition", "first"),
                                    fold=("fold", "first"), n_part=("partition", "nunique")).reset_index()
    assert (out.n_part == 1).all(), "a project was split across partitions"
    return out.drop(columns="n_part")


def apply_assignment(df: pd.DataFrame, assignment: pd.DataFrame) -> pd.DataFrame:
    m = df.merge(assignment[["repo_url", "group_id", "partition", "fold"]], on="repo_url", how="left")
    if m.partition.isna().any():
        raise ValueError("some projects have no assignment; build it from the primary dataset")
    return m


def get_development(df_assigned: pd.DataFrame) -> pd.DataFrame:
    return df_assigned[df_assigned.partition == "dev"].copy()


def get_test(df_assigned: pd.DataFrame, confirm_final_evaluation: bool = False) -> pd.DataFrame:
    """The sealed test set. Only the final, single evaluation may unseal it."""
    if not confirm_final_evaluation:
        raise PermissionError("Test data is sealed. Use it only for the final evaluation "
                              "(confirm_final_evaluation=True), never for development.")
    return df_assigned[df_assigned.partition == "test"].copy()


# --------------------------------------------------------------------------
def _summ(d: pd.DataFrame) -> dict:
    out = {"rows": int(len(d)), "pre_fix_1": int((d.label == 1).sum()),
           "post_fix_0": int((d.label == 0).sum()), "projects": int(d.repo_url.nunique()),
           "commits": int(d.commit_hash.nunique())}
    if "unit" in d.columns:
        out["complete_pairs"] = int((d.groupby("unit").label.nunique() == 2).sum())
    return out


def partition_table(df_assigned: pd.DataFrame) -> dict:
    out = {"test": _summ(df_assigned[df_assigned.partition == "test"]),
           "development": _summ(df_assigned[df_assigned.partition == "dev"]), "folds": {}}
    for k in sorted(df_assigned[df_assigned.partition == "dev"].fold.unique()):
        out["folds"][int(k)] = _summ(df_assigned[(df_assigned.partition == "dev") & (df_assigned.fold == k)])
    return out


def verify(primary: pd.DataFrame, secondary: pd.DataFrame | None, assignment: pd.DataFrame,
           id_col: str = "method_change_id", unit_col: str = "unit") -> dict:
    """Hard checks must all be True. Soft checks are reported for review.
    unit_col is the grouping unit that must never cross partitions (pair unit for the old target,
    file_change_id for the function universe). secondary may be None (single dataset)."""
    p = apply_assignment(primary, assignment)
    s = apply_assignment(secondary, assignment) if secondary is not None else None

    def crosses(d, col):
        return int((d.groupby(col).partition.nunique() > 1).sum())

    def fold_crosses(d, col):
        dev = d[d.partition == "dev"]
        return int((dev.groupby(col).fold.nunique() > 1).sum())

    both = [x for x in (p, s) if x is not None]
    legacy = unit_col == "unit"
    k_part = "no_pair_unit_crosses_partitions" if legacy else f"no_{unit_col}_crosses_partitions"
    k_fold = "no_pair_unit_crosses_folds" if legacy else f"no_{unit_col}_crosses_folds"
    hard = {
        "every_project_in_one_partition": bool((p.groupby("repo_url").partition.nunique() == 1).all()),
        "every_dev_project_in_one_fold": bool((p[p.partition == "dev"].groupby("repo_url").fold.nunique() == 1).all()),
        k_part: all(crosses(d, unit_col) == 0 for d in both),
        k_fold: all(fold_crosses(d, unit_col) == 0 for d in both),
        "no_commit_crosses_partitions": crosses(p, "commit_hash") == 0,
        "no_code_text_crosses_partitions": crosses(p, "code_hash") == 0,
    }
    if s is not None:
        hard["secondary_projects_inside_primary_assignment"] = bool(set(s.repo_url) <= set(p.repo_url))
        hard["secondary_uses_identical_assignment"] = bool(
            (s.merge(p[[id_col, "partition", "fold"]], on=id_col, suffixes=("", "_p"))
              .pipe(lambda m: (m.partition == m.partition_p) & (m.fold == m.fold_p))).all())
    dev = p[p.partition == "dev"]
    min_cell = int(min((dev[(dev.fold == k) & (dev.label == lab)].shape[0]
                        for k in dev.fold.unique() for lab in (0, 1))))
    soft = {"test_row_share": float((p.partition == "test").mean()),
            "min_rows_per_label_per_fold_primary": min_cell,
            "folds_meet_sanity_floor": min_cell >= MIN_PER_LABEL_PER_FOLD}
    return {"hard_checks": hard, "all_hard_checks_pass": all(hard.values()), "soft_checks": soft}


def render(rep: dict) -> str:
    L = ["# Project-grouped partition report", "",
         f"Seed {rep['seed']}; test about 20%; {rep['n_folds']} development folds. " + rep.get("label_note", ""), "",
         "## Verification", ""]
    L += [f"- {k}: **{v}**" for k, v in rep["verification"]["hard_checks"].items()]
    L += [f"- all hard checks pass: **{rep['verification']['all_hard_checks_pass']}**",
          f"- soft: {json.dumps(rep['verification']['soft_checks'])}"]
    for name in ("primary", "secondary"):
        t = rep.get(name)
        if not t:
            continue
        cols = [c for c in ("rows", "pre_fix_1", "post_fix_0", "projects", "commits", "complete_pairs") if c in t["test"]]
        head = {"rows": "Rows", "pre_fix_1": rep.get("label1_name", "Label 1"), "post_fix_0": rep.get("label0_name", "Label 0"),
                "projects": "Projects", "commits": "Commits", "complete_pairs": "Complete pairs"}
        L += ["", f"## {name.capitalize()} dataset", "", "| Partition | " + " | ".join(head[c] for c in cols) + " |",
              "|---|" + "---|" * len(cols)]
        L.append("| test (sealed) | " + " | ".join(f"{t['test'][k]:,}" for k in cols) + " |")
        L.append("| development | " + " | ".join(f"{t['development'][k]:,}" for k in cols) + " |")
        for k, v in t["folds"].items():
            L.append(f"| dev fold {k} | " + " | ".join(f"{v[c]:,}" for c in cols) + " |")
    return "\n".join(L)


def run(universe: bool = False) -> dict:
    from src.data import settings as S
    processed = S.ROOT / "data" / "processed"
    if universe:
        csv = processed / "functions_capped.csv"
        primary = pd.read_csv(csv)
        assignment = build_assignment(primary)
        out_csv, tag = processed / "split_assignment_v2.csv", "_v2"
        assignment.to_csv(out_csv, index=False)
        rep = {"seed": SEED, "n_folds": N_FOLDS, "input_csv": str(csv),
               "label_note": "Label 1 = function changed by the CVE-fixing commit; label 0 = other function in the same "
                             "pre-fix file (not verified vulnerable or safe).",
               "label1_name": "Label 1 (changed)", "label0_name": "Label 0 (unchanged)",
               "primary": partition_table(apply_assignment(primary, assignment)),
               "verification": verify(primary, None, assignment, id_col="function_id", unit_col="file_change_id")}
    else:
        csv = processed / "methods_c_cpp.csv"
        primary = pd.read_csv(csv)
        secondary = primary[primary.pair_complete].copy()
        assignment = build_assignment(primary)
        out_csv, tag = processed / "split_assignment.csv", ""
        assignment.to_csv(out_csv, index=False)
        rep = {"seed": SEED, "n_folds": N_FOLDS, "input_csv": str(csv),
               "label_note": "Pre-fix = label 1, post-fix = label 0.",
               "primary": partition_table(apply_assignment(primary, assignment)),
               "secondary": partition_table(apply_assignment(secondary, assignment)),
               "verification": verify(primary, secondary, assignment)}
    S.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (S.REPORT_DIR / f"split_report{tag}.json").write_text(json.dumps(rep, indent=2, default=str))
    (S.REPORT_DIR / f"split_summary{tag}.md").write_text(render(rep))
    return rep


if __name__ == "__main__":
    import sys
    r = run(universe="--universe" in sys.argv)
    print("All hard checks pass:", r["verification"]["all_hard_checks_pass"])
    print("Written split assignment and report under data/processed and reports")
