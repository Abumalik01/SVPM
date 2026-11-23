"""Phase B, step 2: audit the CVEfixes SQLite database (read-only).

Every number in the output is computed from the database at run time.
Nothing is typed in by hand. If the database is missing the report says
"NOT_ACQUIRED" and contains no statistics.

Run:  python -m src.data.audit
Output: reports/audit_report.json  (machine-readable)
        reports/audit_summary.md   (readable, generated from the JSON)
"""
from __future__ import annotations

import ast
import datetime as dt
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import settings as S

MISSING_TOKENS = {"", "none", "nan", "null"}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def connect_ro(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def q(con, sql, params=()):
    return con.execute(sql, params).fetchall()


def tables(con) -> list[str]:
    return [r[0] for r in q(con, "SELECT name FROM sqlite_master WHERE type='table'")]


def columns(con, table: str) -> list[str]:
    return [r[1] for r in q(con, f'PRAGMA table_info("{table}")')]


def count_params(value) -> float:
    """CVEfixes stores `parameters` as the text of a Python list."""
    if value is None:
        return np.nan
    try:
        parsed = ast.literal_eval(str(value))
        return float(len(parsed)) if isinstance(parsed, (list, tuple)) else np.nan
    except (ValueError, SyntaxError):
        return np.nan


def norm_label(v):
    s = str(v).strip().lower()
    if s == "true":
        return 1
    if s == "false":
        return 0
    return np.nan


def univariate_auc(x: pd.Series, y: pd.Series) -> float | None:
    """Rank-based AUC of one feature vs the label. Descriptive only."""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    n1, n0 = int((d.y == 1).sum()), int((d.y == 0).sum())
    if n1 == 0 or n0 == 0:
        return None
    ranks = d.x.rank(method="average")
    return float((ranks[d.y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# --------------------------------------------------------------------------
# loading the method-level frame (code text is hashed, not kept in memory)
# --------------------------------------------------------------------------
def load_method_frame(con) -> pd.DataFrame:
    """One row per method_change row, joined to its file, commit and project.
    repo_url comes from commits via the file's hash. n_repos_for_hash records
    hashes that appear under more than one repository (forks)."""
    sql = """
    SELECT m.method_change_id, m.file_change_id, m.name, m.signature,
           m.parameters, m.start_line, m.end_line, m.nloc, m.complexity,
           m.token_count, m.top_nesting_level, m.before_change,
           f.hash AS commit_hash, f.filename, f.programming_language,
           h.repo_url, h.n_repos AS n_repos_for_hash
    FROM method_change m
    LEFT JOIN file_change f ON f.file_change_id = m.file_change_id
    LEFT JOIN (SELECT hash, MIN(repo_url) AS repo_url,
                      COUNT(DISTINCT repo_url) AS n_repos
               FROM commits GROUP BY hash) h ON h.hash = f.hash
    """
    df = pd.read_sql_query(sql, con)
    hashes, lens = {}, {}
    cur = con.execute("SELECT method_change_id, code FROM method_change")
    while True:
        batch = cur.fetchmany(5000)
        if not batch:
            break
        for mid, code in batch:
            if code is None:
                hashes[mid], lens[mid] = None, np.nan
            else:
                hashes[mid] = hashlib.md5(code.encode("utf-8", "replace")).hexdigest()
                lens[mid] = len(code)
    df["code_hash"] = df["method_change_id"].map(hashes)
    df["code_len"] = df["method_change_id"].map(lens)
    df["label"] = df["before_change"].map(norm_label)  # 1 = pre-fix, 0 = post-fix
    for c in ["nloc", "complexity", "token_count", "top_nesting_level",
              "start_line", "end_line"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["n_params"] = df["parameters"].map(count_params)
    df["language"] = df["programming_language"].astype("string").str.strip()
    return df


# --------------------------------------------------------------------------
# audit sections
# --------------------------------------------------------------------------
def section_schema(con) -> dict:
    found = tables(con)
    out = {"tables_found": found, "tables": {}}
    for t in found:
        cols = columns(con, t)
        n = q(con, f'SELECT COUNT(*) FROM "{t}"')[0][0]
        exp = S.EXPECTED_SCHEMA.get(t)
        out["tables"][t] = {
            "rows": n, "columns": cols,
            "missing_expected_columns": sorted(set(exp) - set(cols)) if exp else None,
            "extra_columns": sorted(set(cols) - set(exp)) if exp else None,
        }
    out["expected_tables_missing"] = sorted(set(S.EXPECTED_SCHEMA) - set(found))
    return out


def section_labels(df: pd.DataFrame) -> dict:
    raw = df["before_change"].astype("string").value_counts(dropna=False)
    lab = df["label"].value_counts(dropna=False)
    n = len(df)
    n1, n0 = int((df.label == 1).sum()), int((df.label == 0).sum())
    # pairing: same file + same signature present with both labels
    g = df.dropna(subset=["label"]).groupby(["file_change_id", "signature"])["label"]
    sets = g.agg(lambda s: frozenset(s.unique()))
    both = int((sets.map(len) == 2).sum())
    return {
        "method_rows": n,
        "raw_before_change_values": {str(k): int(v) for k, v in raw.items()},
        "label_definition": ("label=1 when before_change=='True' (method as it was "
                             "BEFORE a CVE fix); label=0 when 'False' (same changed "
                             "method AFTER the fix). Source: authors' collection code."),
        "label_1_pre_fix": n1, "label_0_post_fix": n0,
        "unparseable_label_rows": int(df.label.isna().sum()),
        "minority_share": float(min(n1, n0) / (n1 + n0)) if (n1 + n0) else None,
        "unique_file_signature_groups": int(len(sets)),
        "groups_with_both_versions": both,
        "groups_with_only_pre_fix": int(sum(s == {1} for s in sets)),
        "groups_with_only_post_fix": int(sum(s == {0} for s in sets)),
        "_value_counts_check": {str(k): int(v) for k, v in lab.items()},
    }


def section_languages(df: pd.DataFrame, cfg: S.AuditConfig) -> dict:
    d = df.dropna(subset=["label"])
    top = d["language"].value_counts(dropna=False).head(cfg.top_languages)
    per_lang = {}
    wanted = list(top.index.astype(object)) + [
        l for l in S.LANGUAGES_OF_INTEREST if l not in set(top.index.astype(object))]
    for lang in wanted:
        sub = d[d.language == lang] if lang is not None and not pd.isna(lang) \
            else d[d.language.isna()]
        n1, n0 = int((sub.label == 1).sum()), int((sub.label == 0).sum())
        proj = sub.groupby("repo_url").size().sort_values(ascending=False)
        per_lang[str(lang)] = {
            "method_rows": int(len(sub)), "label_1_pre_fix": n1, "label_0_post_fix": n0,
            "projects": int(proj.size),
            "commits": int(sub.commit_hash.nunique()),
            "top_project_share": float(proj.iloc[0] / proj.sum()) if proj.size else None,
            "top5_project_share": float(proj.head(5).sum() / proj.sum()) if proj.size else None,
        }
    # cross-check guessed language against file extension
    ext = d.assign(ext=d.filename.astype("string").str.extract(r"(\.[^.]+)$")[0].str.lower())
    agree = {}
    for lang, allowed in S.EXT_EXPECTED.items():
        sub = ext[ext.language == lang]
        if len(sub):
            agree[lang] = {"rows": int(len(sub)),
                           "extension_matches_expected": float(sub.ext.isin(allowed).mean())}
    return {"note": ("programming_language is GUESSED by the 'guesslang' tool on the "
                     "post-fix file content (authors' code), not read from the file "
                     "extension. Agreement with extension is shown below."),
            "method_rows_by_language_top": {str(k): int(v) for k, v in top.items()},
            "per_language": per_lang, "extension_agreement": agree}


def section_missing(con, df: pd.DataFrame) -> dict:
    out = {"method_change": {}, "invalid_numeric": {}}
    for col in S.EXPECTED_SCHEMA["method_change"]:
        if col not in columns(con, "method_change"):
            continue
        nulls = q(con, f'SELECT COUNT(*) FROM method_change WHERE "{col}" IS NULL')[0][0]
        textlike = q(con, f"""SELECT COUNT(*) FROM method_change WHERE "{col}" IS NOT NULL
                              AND LOWER(TRIM(CAST("{col}" AS TEXT))) IN ('', 'none', 'nan', 'null')""")[0][0]
        out["method_change"][col] = {"null": int(nulls), "empty_or_none_text": int(textlike)}
    out["invalid_numeric"] = {
        "nloc_le_0": int((df.nloc <= 0).sum()),
        "complexity_lt_1": int((df.complexity < 1).sum()),
        "token_count_le_0": int((df.token_count <= 0).sum()),
        "negative_nesting": int((df.top_nesting_level < 0).sum()),
        "end_line_before_start_line": int((df.end_line < df.start_line).sum()),
        "parameters_unparseable": int(df.n_params.isna().sum()),
    }
    out["file_change_programming_language_null"] = q(
        con, "SELECT COUNT(*) FROM file_change WHERE programming_language IS NULL")[0][0]
    return out


def section_duplicates(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=["label"])
    key = ["file_change_id", "signature", "start_line", "end_line", "label"]
    dup_rows = int(d.duplicated(subset=key, keep="first").sum())
    by_code = d.dropna(subset=["code_hash"]).groupby("code_hash")
    sizes = by_code.size()
    labels = by_code["label"].nunique()
    repos = by_code["repo_url"].nunique()
    # within a before/after pair, is the code text identical?
    pair = d.dropna(subset=["code_hash"]).groupby(["file_change_id", "signature"])
    same_text_conflict = int(sum(
        1 for _, g in pair if g.label.nunique() == 2 and g.code_hash.nunique() == 1))
    return {
        "duplicate_method_change_id": int(df.method_change_id.duplicated().sum()),
        "duplicate_rows_same_file_signature_lines_label": dup_rows,
        "distinct_code_texts": int(sizes.size),
        "code_texts_appearing_more_than_once": int((sizes > 1).sum()),
        "rows_in_repeated_code_texts": int(sizes[sizes > 1].sum()),
        "code_texts_with_both_labels (label conflict)": int((labels > 1).sum()),
        "code_texts_spanning_more_than_one_project": int((repos > 1).sum()),
        "pairs_with_identical_code_text_but_both_labels": same_text_conflict,
    }


def section_features(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=["label"])
    stats = {}
    for col in S.NUMERIC_FEATURES:
        s = d[col]
        stats[col] = {
            "non_null": int(s.notna().sum()),
            "by_label": {
                lbl: {"mean": _f(g.mean()), "median": _f(g.median()),
                      "p95": _f(g.quantile(0.95)), "max": _f(g.max())}
                for lbl, g in (("pre_fix_1", s[d.label == 1]), ("post_fix_0", s[d.label == 0]))
            },
            "univariate_auc_descriptive": univariate_auc(s, d.label),
        }
    # before/after pair differences
    base = ["nloc", "complexity", "token_count", "top_nesting_level", "n_params"]
    g = d.groupby(["file_change_id", "signature", "label"])[base].first().unstack("label")
    g = g.dropna()
    if len(g):
        same = np.ones(len(g), dtype=bool)
        for c in base:
            same &= (g[(c, 1)].values == g[(c, 0)].values)
        pairs = {"matched_pairs": int(len(g)),
                 "share_with_all_metrics_identical": float(same.mean())}
    else:
        pairs = {"matched_pairs": 0, "share_with_all_metrics_identical": None}
    return {"note": ("univariate AUC is a descriptive statistic of one feature "
                     "against the label on ALL rows; it is NOT a model result and "
                     "ignores project grouping."),
            "numeric_features": stats, "before_after_pair_diagnostic": pairs}


def section_relationships(con, df: pd.DataFrame) -> dict:
    def one(sql):
        return q(con, sql)[0][0]
    orphan = {
        "method_without_file": one("""SELECT COUNT(*) FROM method_change m LEFT JOIN file_change f
                                      ON f.file_change_id=m.file_change_id WHERE f.file_change_id IS NULL"""),
        "file_without_commit": one("""SELECT COUNT(*) FROM file_change f LEFT JOIN commits c
                                      ON c.hash=f.hash WHERE c.hash IS NULL"""),
        "commit_without_fix_record": one("""SELECT COUNT(*) FROM commits c LEFT JOIN fixes x
                                            ON x.hash=c.hash WHERE x.hash IS NULL"""),
        "fix_without_cve": one("""SELECT COUNT(*) FROM fixes x LEFT JOIN cve v
                                  ON v.cve_id=x.cve_id WHERE v.cve_id IS NULL"""),
        "commit_without_repository": one("""SELECT COUNT(*) FROM commits c LEFT JOIN repository r
                                            ON r.repo_url=c.repo_url WHERE r.repo_url IS NULL"""),
    }
    d = df.dropna(subset=["label"])
    per_proj = d.groupby("repo_url").size().sort_values(ascending=False)
    per_commit = d.groupby("commit_hash").size()
    return {
        "orphans": orphan,
        "hashes_under_multiple_repositories": int((df.n_repos_for_hash > 1).sum()),
        "distinct_projects_with_method_rows": int(per_proj.size),
        "distinct_commits_with_method_rows": int(per_commit.size),
        "cves_per_commit_max": one("SELECT MAX(c) FROM (SELECT COUNT(DISTINCT cve_id) c FROM fixes GROUP BY hash)"),
        "methods_per_project": {"median": _f(per_proj.median()), "max": _f(per_proj.max()),
                                "top10_share": _f(per_proj.head(10).sum() / per_proj.sum()) if per_proj.size else None},
        "methods_per_commit": {"median": _f(per_commit.median()), "max": _f(per_commit.max())},
        "grouping_unit_recommendation": "repo_url (project) as the group for splitting; "
                                        "commit_hash is the finer alternative.",
    }


def section_leakage_register() -> list[dict]:
    """Static risk register. The reasons come from how the authors' collection
    code builds each column; availability is a judgement for review."""
    R = lambda f, why, avail, use: {"field": f, "risk": why, "available_at_prediction_time": avail, "proposed_use": use}
    return [
        R("method_change.before_change", "IS the label", False, "target only"),
        R("method_change.code (text)", "free text; identifiers/comments may encode project or fix hints", True, "excluded from metric-based features"),
        R("method_change.nloc / complexity / token_count / top_nesting_level / parameters", "computed by Lizard on the method alone", True, "CANDIDATE FEATURES"),
        R("method_change.signature / name / start_line / end_line / *_id", "identifiers or positions; can memorise projects", True, "grouping/pairing keys only, not features"),
        R("file_change.nloc / complexity / token_count", "computed on POST-fix file content only (authors' code uses file.source_code)", False, "exclude (post-fix information)"),
        R("file_change.diff / diff_parsed / num_lines_added / num_lines_deleted / change_type", "exist only because a fix happened", False, "exclude"),
        R("file_change.code_after", "post-fix code", False, "exclude"),
        R("file_change.programming_language", "guessed from post-fix code", True, "subset filter only"),
        R("commits.* (msg, num_lines_*, dmm_*, merge, parents, dates)", "describe the fixing commit", False, "exclude"),
        R("fixes.*, cve.*, cwe.*, cwe_classification.*", "CVE ids, severity, CWE type reveal the vulnerability", False, "exclude (analysis/stratified reporting only)"),
        R("repository.* (stars, forks, dates, owner)", "project-level; identifies project", True, "exclude from features; project id used as split group"),
    ]


def _f(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


# --------------------------------------------------------------------------
# recommendation (rule-based, advisory)
# --------------------------------------------------------------------------
def section_recommendation(report: dict, cfg: S.AuditConfig) -> dict:
    r = cfg.rules
    langs = report["languages"]["per_language"]
    pairs = report["features"]["before_after_pair_diagnostic"]
    rows = []
    for lang, st in langs.items():
        minority = min(st["label_1_pre_fix"], st["label_0_post_fix"])
        checks = {
            "enough_projects": st["projects"] >= r.min_projects,
            "enough_minority_rows": minority >= r.min_minority_rows,
            "no_dominant_project": (st["top_project_share"] or 1) <= r.max_top_project_share,
        }
        rows.append({"language": lang, "minority_rows": minority, "projects": st["projects"],
                     "top_project_share": st["top_project_share"], "checks": checks,
                     "passes_all": all(checks.values())})
    passing = sorted([x for x in rows if x["passes_all"]], key=lambda x: -x["minority_rows"])
    n_files = report["schema"]["tables"].get("file_change", {}).get("rows")
    n_methods = report["schema"]["tables"].get("method_change", {}).get("rows")
    exact = pairs.get("share_with_all_metrics_identical")
    return {
        "status": "ADVISORY - for student and supervisor review",
        "rules_used": vars(r),
        "language_candidates": rows,
        "suggested_primary_language": passing[0]["language"] if passing else None,
        "suggested_multi_language_set": [x["language"] for x in passing],
        "granularity": {
            "suggested": "method",
            "reasons": [
                f"method_change has {n_methods} rows vs file_change {n_files} rows (more observations).",
                "file_change.nloc/complexity/token_count are computed on POST-fix files only, "
                "so a file-level before/after label would need Lizard re-run on code_before "
                "(extra work, extra risk).",
                "method_change stores Lizard metrics for BOTH the pre-fix and post-fix version "
                "of each changed method, which gives a direct, defensible label.",
            ],
        },
        "signal_warning": (
            None if exact is None else
            (f"{exact:.1%} of matched before/after method pairs have identical values for all "
             "stored metrics. High values mean metric-only models cannot separate the two "
             "labels; consider the supervisor discussion in 'open_questions'.")
            if exact > r.max_exact_pair_share else
            f"{exact:.1%} of matched pairs have identical metrics (below the advisory limit)."),
    }


OPEN_QUESTIONS = [
    "Label semantics: CVEfixes keeps only methods CHANGED by a vulnerability-fixing commit. "
    "label 0 is the post-fix version of the same method; it is not a randomly sampled 'clean' method, "
    "and post-fix code is not guaranteed secure.",
    "Some changed methods in a fix commit may be unrelated to the vulnerability (refactoring, tests).",
    "Language is guessed by a tool on post-fix content; mislabelled files are possible.",
    "Both versions of one method must stay in the same split (group by project or commit).",
    "Generalisation to unseen projects can only be estimated with project-grouped splitting.",
    "Method metrics describe size and complexity; they may carry weak security signal. This is a "
    "research finding to report honestly, not a defect to hide.",
]


# --------------------------------------------------------------------------
# orchestration + summary rendering
# --------------------------------------------------------------------------
def run_audit(cfg: S.AuditConfig | None = None) -> dict:
    cfg = cfg or S.AuditConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    report = {"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "database": str(cfg.db_path)}
    if not Path(cfg.db_path).exists():
        report["status"] = "NOT_ACQUIRED"
        report["message"] = (f"{cfg.db_path} does not exist. No statistics were computed. "
                             "Run: python -m src.data.acquire --all")
    else:
        con = connect_ro(cfg.db_path)
        try:
            report["status"] = "AUDITED"
            report["schema"] = section_schema(con)
            missing_tables = report["schema"]["expected_tables_missing"]
            if "method_change" in missing_tables or "file_change" in missing_tables:
                report["status"] = "SCHEMA_MISMATCH"
            else:
                df = load_method_frame(con)
                report["labels"] = section_labels(df)
                report["languages"] = section_languages(df, cfg)
                report["missing_values"] = section_missing(con, df)
                report["duplicates"] = section_duplicates(df)
                report["features"] = section_features(df)
                report["relationships"] = section_relationships(con, df)
                report["leakage_register"] = section_leakage_register()
                report["recommendation"] = section_recommendation(report, cfg)
                report["open_questions"] = OPEN_QUESTIONS
        finally:
            con.close()
    (cfg.out_dir / "audit_report.json").write_text(json.dumps(report, indent=2, default=str))
    (cfg.out_dir / "audit_summary.md").write_text(render_summary(report))
    return report


def _pct(x):
    return "n/a" if x is None else f"{x:.1%}"


def render_summary(rep: dict) -> str:
    L = ["# CVEfixes audit summary", "", f"Generated (UTC): {rep['generated_utc']}",
         f"Status: **{rep['status']}**", ""]
    if rep["status"] == "NOT_ACQUIRED":
        L += [rep["message"], "", "No dataset statistics exist yet."]
        return "\n".join(L)
    if rep["status"] == "SCHEMA_MISMATCH":
        L += ["Expected tables are missing:", str(rep["schema"]["expected_tables_missing"])]
        return "\n".join(L)
    sc = rep["schema"]["tables"]
    L += ["## 1. Tables (rows)", "", "| Table | Rows |", "|---|---|"]
    L += [f"| {t} | {v['rows']:,} |" for t, v in sc.items()]
    lb = rep["labels"]
    L += ["", "## 2. Target label", "", lb["label_definition"], "",
          f"- pre-fix (1): {lb['label_1_pre_fix']:,}", f"- post-fix (0): {lb['label_0_post_fix']:,}",
          f"- unparseable: {lb['unparseable_label_rows']:,}",
          f"- minority share: {_pct(lb['minority_share'])}",
          f"- file+signature groups with both versions: {lb['groups_with_both_versions']:,} "
          f"(only pre-fix: {lb['groups_with_only_pre_fix']:,}; only post-fix: {lb['groups_with_only_post_fix']:,})"]
    L += ["", "## 3. Languages (method level)", "", rep["languages"]["note"], "",
          "| Language | Rows | Pre-fix | Post-fix | Projects | Top-project share |", "|---|---|---|---|---|---|"]
    for k, v in rep["languages"]["per_language"].items():
        L.append(f"| {k} | {v['method_rows']:,} | {v['label_1_pre_fix']:,} | {v['label_0_post_fix']:,} | "
                 f"{v['projects']:,} | {_pct(v['top_project_share'])} |")
    L += ["", "Extension agreement: " + json.dumps(rep["languages"]["extension_agreement"])]
    mv = rep["missing_values"]
    L += ["", "## 4. Missing and invalid values", "", json.dumps(mv["invalid_numeric"]),
          "", "Per-column NULL/empty counts are in the JSON report."]
    L += ["", "## 5. Duplicates", ""] + [f"- {k}: {v:,}" for k, v in rep["duplicates"].items()]
    L += ["", "## 6. Features (descriptive only)", "", rep["features"]["note"], "",
          "| Feature | Univariate AUC |", "|---|---|"]
    for k, v in rep["features"]["numeric_features"].items():
        a = v["univariate_auc_descriptive"]
        L.append(f"| {k} | {'n/a' if a is None else f'{a:.3f}'} |")
    pdg = rep["features"]["before_after_pair_diagnostic"]
    L += ["", f"Matched before/after pairs: {pdg['matched_pairs']:,}; all metrics identical: "
              f"{_pct(pdg['share_with_all_metrics_identical'])}"]
    rl = rep["relationships"]
    L += ["", "## 7. Relationships", "", f"- orphans: {json.dumps(rl['orphans'])}",
          f"- hashes under multiple repositories: {rl['hashes_under_multiple_repositories']:,}",
          f"- projects: {rl['distinct_projects_with_method_rows']:,}; commits: {rl['distinct_commits_with_method_rows']:,}",
          f"- top-10 projects' share of method rows: {_pct(rl['methods_per_project']['top10_share'])}"]
    L += ["", "## 8. Leakage register", "", "| Field | Risk | Known at prediction? | Proposed use |", "|---|---|---|---|"]
    for r in rep["leakage_register"]:
        L.append(f"| {r['field']} | {r['risk']} | {r['available_at_prediction_time']} | {r['proposed_use']} |")
    rc = rep["recommendation"]
    L += ["", "## 9. Advisory recommendation", "", f"Granularity: **{rc['granularity']['suggested']}**"]
    L += [f"- {x}" for x in rc["granularity"]["reasons"]]
    L += ["", f"Suggested primary language: **{rc['suggested_primary_language']}**",
          f"Languages passing all advisory checks: {rc['suggested_multi_language_set']}",
          f"Rules used: {json.dumps(rc['rules_used'])}", "", f"Signal check: {rc['signal_warning']}",
          "", "## 10. Open questions for review", ""] + [f"- {x}" for x in rep["open_questions"]]
    return "\n".join(L)


def main() -> int:
    rep = run_audit()
    print("Audit status:", rep["status"])
    print("Reports written to", S.REPORT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
