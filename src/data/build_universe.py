"""Target v2, step 1: build the C/C++ function universe from CVEfixes (NO model training here).

Label definition (operational, not independently verified vulnerable/safe code):
  label 1 = a function in a PRE-FIX file that the CVE-fixing commit changed;
  label 0 = any other function in the same pre-fix file (not changed by the fix).
Only pre-fix code (file_change.code_before) is used; no post-fix code enters the dataset.

Rules (all counts are reported, nothing is silently dropped):
  * files: guessed language C or C++, non-empty code_before, at least one changed method.
  * every function is extracted with Lizard from the whole pre-fix file (same process as training
    and as the future prediction on uploaded files).
  * a changed method (method_change.before_change = 'True') is matched to a Lizard function by
    (name, start_line), else by a unique name. In STRICT mode (default) a file with any unmatched
    changed method is dropped, so a changed function is never silently labelled 0.
  * files over MAX_FILE_CHARS are skipped (reported).
Cleaning afterwards (reusing subset_audit helpers): remove function texts carrying both labels,
remove texts shared by several projects, keep one copy of repeats inside a project.
Caps are NOT applied here; they are decided after reading the report.

Run:  python -m src.data.build_universe            (add --limit 200 for a quick trial)
Output: data/interim/function_universe.csv, reports/universe_report.json, universe_summary.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data import subset_audit as sa

MAX_FILE_CHARS = 400_000
FEATURES = ["nloc", "complexity", "token_count", "top_nesting_level", "n_params"]


# --------------------------------------------------------------------------
def lizard_functions(filename: str, source: str) -> list[dict]:
    import lizard
    info = lizard.analyze_file.analyze_source_code(filename, source)
    return [{"name": f.name, "start_line": int(f.start_line), "end_line": int(f.end_line),
             "nloc": f.nloc, "complexity": f.cyclomatic_complexity, "token_count": f.token_count,
             "top_nesting_level": f.top_nesting_level, "n_params": f.parameter_count}
            for f in info.function_list]


def match_changed(funcs: list[dict], changed: list[tuple]) -> tuple[set, int]:
    """Return (indexes of matched functions, number of unmatched changed rows)."""
    hit, unmatched = set(), 0
    for name, start in changed:
        exact = [i for i, f in enumerate(funcs) if f["name"] == name and f["start_line"] == start]
        same = [i for i, f in enumerate(funcs) if f["name"] == name]
        pick = exact[0] if exact else (same[0] if len(same) == 1 else None)
        if pick is None:
            unmatched += 1
        else:
            hit.add(pick)
    return hit, unmatched


def function_text(source: str, start: int, end: int) -> str:
    return "\n".join(source.splitlines()[start - 1:end])


def label_file(filename: str, source: str, changed: list[tuple], strict: bool = True):
    """Return (status, rows). rows carry label and features for every function in the file."""
    if len(source) > MAX_FILE_CHARS:
        return "too_large", [], 0
    try:
        funcs = lizard_functions(filename, source)
    except Exception:                                    # noqa: BLE001 - a parser failure is a finding
        return "lizard_error", [], 0
    if not funcs:
        return "no_functions", [], 0
    hit, unmatched = match_changed(funcs, changed)
    if unmatched and strict:
        return "dropped_unmatched_changed", [], unmatched
    rows = []
    for i, f in enumerate(funcs):
        h = hashlib.md5(function_text(source, f["start_line"], f["end_line"]).encode("utf-8", "replace")).hexdigest()
        rows.append({**f, "label": int(i in hit), "code_hash": h, "n_changed_in_file": len(hit)})
    return "ok", rows, unmatched


# --------------------------------------------------------------------------
def build_function_rows(con: sqlite3.Connection, limit: int | None = None, strict: bool = True,
                        progress_every: int = 500):
    repo = dict(con.execute("SELECT hash, MIN(repo_url) FROM commits GROUP BY hash").fetchall())
    changed: dict = {}
    for fid, name, start in con.execute("""SELECT file_change_id, name, start_line FROM method_change
            WHERE LOWER(TRIM(CAST(before_change AS TEXT))) = 'true'"""):
        changed.setdefault(fid, []).append((name, int(float(start))))
    sql = """SELECT f.file_change_id, f.hash, f.filename, f.programming_language, f.code_before
             FROM file_change f
             WHERE f.programming_language IN ('C', 'C++') AND f.code_before IS NOT NULL
               AND TRIM(f.code_before) <> ''
               AND EXISTS (SELECT 1 FROM method_change m WHERE m.file_change_id = f.file_change_id
                           AND LOWER(TRIM(CAST(m.before_change AS TEXT))) = 'true')
             ORDER BY f.file_change_id"""
    if limit:
        sql += f" LIMIT {int(limit)}"
    total_candidates = con.execute("""SELECT COUNT(*) FROM file_change f
             WHERE f.programming_language IN ('C', 'C++') AND f.code_before IS NOT NULL
               AND TRIM(f.code_before) <> '' AND EXISTS (SELECT 1 FROM method_change m
               WHERE m.file_change_id = f.file_change_id
               AND LOWER(TRIM(CAST(m.before_change AS TEXT))) = 'true')""").fetchone()[0]
    out, status_counts = [], {}
    changed_total = unmatched_total = 0
    for n, (fid, h, fname, lang, src) in enumerate(con.execute(sql), 1):
        ch = changed.get(fid, [])
        status, rows, unmatched = label_file(fname, src, ch, strict)
        status_counts[status] = status_counts.get(status, 0) + 1
        changed_total += len(ch)
        unmatched_total += unmatched
        for r in rows:
            r.update(file_change_id=fid, commit_hash=h, repo_url=repo.get(h), filename=fname, language=lang)
            out.append(r)
        if progress_every and n % progress_every == 0:
            print(f"  processed {n:,} files, {len(out):,} functions so far ...", flush=True)
    df = pd.DataFrame(out)
    df.insert(0, "function_id", range(len(df)))
    stats = {"candidate_files_in_database": int(total_candidates), "files_processed": int(sum(status_counts.values())),
             "file_status": status_counts, "changed_method_rows_seen": changed_total,
             "changed_method_rows_unmatched": unmatched_total, "strict_mode": strict}
    return df, stats


# --------------------------------------------------------------------------
def summarise(d: pd.DataFrame) -> dict:
    n1, n0 = int((d.label == 1).sum()), int((d.label == 0).sum())
    proj = d.groupby("repo_url").size().sort_values(ascending=False)
    pos_proj = d[d.label == 1].groupby("repo_url").size().sort_values(ascending=False)
    return {"functions": int(len(d)), "label_1_changed": n1, "label_0_unchanged": n0,
            "positive_share": float(n1 / (n1 + n0)) if (n1 + n0) else None,
            "projects": int(proj.size), "projects_with_positives": int(pos_proj.size),
            "files": int(d.file_change_id.nunique()), "commits": int(d.commit_hash.nunique()),
            "top_project_share": float(proj.iloc[0] / proj.sum()) if proj.size else None,
            "top_project_share_of_positives": float(pos_proj.iloc[0] / pos_proj.sum()) if pos_proj.size else None}


def drop_within_project_duplicates(d: pd.DataFrame) -> pd.DataFrame:
    return d.sort_values("function_id").drop_duplicates(subset=["repo_url", "code_hash", "label"], keep="first").copy()


def clean_universe(df: pd.DataFrame):
    steps = [("0 all extracted functions", summarise(df))]
    d = sa.drop_label_conflicts(df)
    steps.append(("1 remove function texts carrying both labels", summarise(d)))
    d = sa.drop_cross_project_duplicates(d)
    steps.append(("2 remove texts shared by several projects", summarise(d)))
    d = drop_within_project_duplicates(d)
    steps.append(("3 keep one copy of repeats inside a project", summarise(d)))
    return steps, d


def feature_description(d: pd.DataFrame) -> dict:
    out = {}
    for f in FEATURES:
        pos, neg = d[d.label == 1][f], d[d.label == 0][f]
        auc = float(roc_auc_score(d.label, d[f])) if d.label.nunique() == 2 and d[f].nunique() > 1 else None
        out[f] = {"median_label_1": float(pos.median()) if len(pos) else None,
                  "median_label_0": float(neg.median()) if len(neg) else None,
                  "univariate_auc_descriptive": auc}
    return out


def label_noise_indicators(d: pd.DataFrame) -> dict:
    pos = d[d.label == 1]
    per_commit = pos.groupby("commit_hash").size()
    big = per_commit[per_commit > 50]
    return {"changed_functions_per_commit": {"median": float(per_commit.median()) if len(per_commit) else None,
                                              "max": int(per_commit.max()) if len(per_commit) else None},
            "commits_with_more_than_50_changed_functions": int(len(big)),
            "share_of_positives_from_those_commits": float(big.sum() / per_commit.sum()) if len(per_commit) else None,
            "note": "Large commits may include cleanup or refactoring, which would make some label-1 functions "
                    "not vulnerability-related. Review before choosing caps."}


def render(rep: dict) -> str:
    pct = lambda x: "n/a" if x is None else f"{x:.1%}"
    L = ["# Function universe report (target v2)", "",
         "Label 1 = function changed by the CVE-fixing commit (pre-fix code). Label 0 = other function in the same "
         "pre-fix file. These are not independently verified vulnerable and safe functions.", "",
         f"Lizard {rep['lizard_version']}; strict mode: {rep['extraction']['strict_mode']}; "
         f"max file size {rep['max_file_chars']:,} characters; seconds: {rep['seconds']}", "",
         "## Files", "", f"- candidate files in database (C/C++, pre-fix code, at least one changed method): "
         f"{rep['extraction']['candidate_files_in_database']:,}",
         f"- files processed: {rep['extraction']['files_processed']:,}"]
    L += [f"- {k}: {v:,}" for k, v in rep["extraction"]["file_status"].items()]
    e = rep["extraction"]
    L += [f"- changed method rows seen: {e['changed_method_rows_seen']:,}; unmatched to a Lizard function: "
          f"{e['changed_method_rows_unmatched']:,}", "", "## Attrition", "",
          "| Step | Functions | Label 1 | Label 0 | Positive share | Projects | Files | Top project share |", "|---|---|---|---|---|---|---|---|"]
    for name, s in rep["attrition"]:
        L.append(f"| {name} | {s['functions']:,} | {s['label_1_changed']:,} | {s['label_0_unchanged']:,} | "
                 f"{pct(s['positive_share'])} | {s['projects']:,} | {s['files']:,} | {pct(s['top_project_share'])} |")
    f = rep["final"]
    L += ["", "## Cleaned universe", "", f"- projects with positives: {f['projects_with_positives']:,}; commits: {f['commits']:,}",
          f"- top project share of all functions: {pct(f['top_project_share'])}; of positives: {pct(f['top_project_share_of_positives'])}",
          "", "Language counts: " + json.dumps(rep["language_counts"]), "",
          "## Features (descriptive only, not a model result)", "", "| Feature | Median label 1 | Median label 0 | Univariate AUC |", "|---|---|---|---|"]
    for k, v in rep["features"].items():
        a = v["univariate_auc_descriptive"]
        L.append(f"| {k} | {v['median_label_1']} | {v['median_label_0']} | {'n/a' if a is None else f'{a:.3f}'} |")
    n = rep["label_noise"]
    L += ["", "## Label-noise indicators", "", json.dumps(n, indent=1), "", f"CSV sha256: `{rep['csv_sha256']}`", ""]
    return "\n".join(L)


def run(limit: int | None = None, strict: bool = True) -> dict:
    import lizard
    from src.data import audit, settings as S
    t0 = time.time()
    con = audit.connect_ro(S.DB_PATH)
    try:
        df, stats = build_function_rows(con, limit=limit, strict=strict)
    finally:
        con.close()
    if df.empty:
        raise SystemExit("No functions were extracted; check the database and the language filter.")
    steps, clean = clean_universe(df)
    suffix = "" if limit is None else "_trial"
    out_csv = S.INTERIM_DIR / f"function_universe{suffix}.csv"
    S.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_csv, index=False)
    rep = {"lizard_version": lizard.version, "max_file_chars": MAX_FILE_CHARS, "limit": limit,
           "extraction": stats, "attrition": steps, "final": steps[-1][1],
           "language_counts": {k: int(v) for k, v in clean.language.value_counts().items()},
           "features": feature_description(clean), "label_noise": label_noise_indicators(clean),
           "csv_path": str(out_csv), "csv_sha256": sa_sha(out_csv), "seconds": round(time.time() - t0, 1)}
    S.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (S.REPORT_DIR / f"universe_report{suffix}.json").write_text(json.dumps(rep, indent=2, default=str))
    (S.REPORT_DIR / f"universe_summary{suffix}.md").write_text(render(rep))
    return rep


def sa_sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process only the first N files (trial)")
    ap.add_argument("--not-strict", action="store_true", help="keep files with unmatched changed methods")
    a = ap.parse_args()
    r = run(a.limit, strict=not a.not_strict)
    print("Done. Functions after cleaning:", r["final"]["functions"], "| positive share:", r["final"]["positive_share"])
