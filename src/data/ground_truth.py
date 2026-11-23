"""Build a manual ground-truth review queue from CVEfixes evidence.

CVEfixes associations are not considered independently verified ground truth.
This module creates candidates for human verification and refuses to export them
as vulnerable or safe until review metadata and evidence are supplied.

Run: python -m src.data.ground_truth
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data import settings as S
from src.data.build_universe import function_text, label_file

VALID_STATUS = {"not_reviewed", "vulnerable_verified", "safe_verified", "not_vulnerable", "invalid", "needs_review"}


def connect_ro(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)


def build_review_queue(db_path: Path = S.DB_PATH, limit: int | None = None) -> pd.DataFrame:
    """Create function-level review candidates; neither class is auto-verified."""
    con = connect_ro(db_path)
    try:
        cwe = con.execute("""
            SELECT cve_id, group_concat(cwe_id, '; ') FROM cwe_classification
            GROUP BY cve_id
        """).fetchall()
        cwe_by_cve = dict(cwe)

        sql = """
            SELECT f.file_change_id, f.hash, x.repo_url, f.filename,
                   f.programming_language, f.code_before, f.diff, x.cve_id,
                   c.description, c.cvss3_base_severity, c.cvss3_base_score, cm.msg
            FROM file_change f
            JOIN fixes x ON x.hash = f.hash
            LEFT JOIN cve c ON c.cve_id = x.cve_id
            LEFT JOIN commits cm ON cm.hash = f.hash AND cm.repo_url = x.repo_url
            WHERE f.programming_language IN ('C', 'C++')
              AND f.code_before IS NOT NULL AND TRIM(f.code_before) <> ''
              AND EXISTS (SELECT 1 FROM method_change m WHERE m.file_change_id = f.file_change_id
                          AND LOWER(TRIM(CAST(m.before_change AS TEXT))) = 'true')
            ORDER BY f.file_change_id
        """
        files = con.execute(sql + (" LIMIT ?" if limit else ""), (limit,) if limit else ()).fetchall()
        changed_by_file: dict[int, list[tuple[str, int]]] = {}
        for file_id, name, start in con.execute("""
            SELECT file_change_id, name, start_line FROM method_change
            WHERE LOWER(TRIM(CAST(before_change AS TEXT))) = 'true'
        """):
            try:
                changed_by_file.setdefault(int(file_id), []).append((str(name), int(float(start))))
            except (TypeError, ValueError):
                continue

        rows = []
        for (file_id, commit, repo, filename, language, source, diff, cve_id,
             cve_desc, severity, cvss, commit_message) in files:
            status, funcs, _ = label_file(filename, source, changed_by_file.get(int(file_id), []), strict=True)
            if status != "ok":
                continue
            for f in funcs:
                start, end, name = int(f["start_line"]), int(f["end_line"]), str(f["name"])
                changed = bool(f["label"])
                rows.append({
                    "candidate_id": f"function:{file_id}:{start}:{name}",
                    "candidate_type": "changed_function" if changed else "unchanged_function_in_fix_file",
                    "function_id": f"{file_id}:{start}:{name}", "file_change_id": file_id,
                    "commit_hash": commit, "repo_url": repo, "filename": filename,
                    "language": language, "function_name": name, "signature": "",
                    "start_line": start, "end_line": end,
                    **{feature: f[feature] for feature in ("nloc", "complexity", "token_count",
                                                          "top_nesting_level", "n_params")},
                    "code_before": function_text(source, start, end),
                    "cve_id": cve_id if changed else "",
                    "cwe_ids": cwe_by_cve.get(cve_id, "") if changed else "",
                    "severity": severity if changed else "", "cvss3_base_score": cvss if changed else "",
                    "commit_message": commit_message if changed else "",
                    "cve_description": cve_desc if changed else "", "diff": diff if changed else "",
                    "ground_truth_status": "not_reviewed", "reviewer": "",
                    "evidence_url": "", "evidence_note": "", "reviewed_at": "",
                    "review_scope": "", "review_method": "", "limitations": "",
                })
    finally:
        con.close()

    return pd.DataFrame(rows).drop_duplicates("candidate_id", keep="first")


def validate_review_queue(df: pd.DataFrame) -> dict:
    """Validate reviews; verified rows require reviewer and evidence or an explicit check note."""
    if df.empty:
        return {"valid": False, "verified_vulnerable": 0, "verified_safe": 0,
                "invalid_statuses": [], "missing_evidence": []}

    invalid_statuses = sorted(set(df["ground_truth_status"].fillna("")) - VALID_STATUS)
    verified = df[df["ground_truth_status"].isin({"vulnerable_verified", "safe_verified"})]
    scope_fields = ["review_scope", "review_method", "limitations"]
    missing_scope = []
    for field in scope_fields:
        if field in verified.columns:
            missing_scope.extend(verified.loc[verified[field].fillna("").astype(str).str.strip().eq(""), "candidate_id"].tolist())
    missing = verified[
        verified["reviewer"].fillna("").str.strip().eq("") |
        (
            verified["evidence_url"].fillna("").str.strip().eq("") &
            verified["evidence_note"].fillna("").str.strip().eq("")
        )
    ]["candidate_id"].tolist()
    return {
        "valid": not invalid_statuses and not missing and not missing_scope,
        "verified_vulnerable": int((df["ground_truth_status"] == "vulnerable_verified").sum()),
        "verified_safe": int((df["ground_truth_status"] == "safe_verified").sum()),
        "not_reviewed": int((df["ground_truth_status"] == "not_reviewed").sum()),
        "invalid_statuses": invalid_statuses,
        "missing_evidence": missing,
        "missing_scope": sorted(set(missing_scope)),
    }


def merge_reviews(template: pd.DataFrame, review_path: Path | pd.DataFrame) -> pd.DataFrame:
    reviews = review_path.copy() if isinstance(review_path, pd.DataFrame) else pd.read_csv(review_path)
    required = {"candidate_id", "ground_truth_status", "reviewer", "evidence_url", "evidence_note", "reviewed_at",
                "review_scope", "review_method", "limitations"}
    missing = sorted(required - set(reviews.columns))
    if missing:
        raise ValueError("review CSV missing columns: " + ", ".join(missing))
    keep = reviews[["candidate_id", *sorted(required - {"candidate_id"})]].drop_duplicates("candidate_id")
    out = template.drop(columns=[c for c in keep.columns if c != "candidate_id"], errors="ignore").merge(keep, on="candidate_id", how="left")
    for c in ("ground_truth_status", "reviewer", "evidence_url", "evidence_note", "reviewed_at",
              "review_scope", "review_method", "limitations"):
        if c in out.columns:
            out[c] = out[c].fillna(template[c] if c in template.columns else "")
    return out


def export_verified(df: pd.DataFrame, path: Path) -> dict:
    validation = validate_review_queue(df)
    if not validation["valid"]:
        raise ValueError("Ground truth review is incomplete: " + json.dumps(validation))
    verified = df[df["ground_truth_status"].isin({"vulnerable_verified", "safe_verified"})].copy()
    verified["ground_truth_label"] = verified["ground_truth_status"].map({
        "vulnerable_verified": 1, "safe_verified": 0,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    verified.to_csv(path, index=False)
    return validation


def run(db_path: Path = S.DB_PATH, limit: int | None = None, review_path: Path | None = None,
        verified_output: Path | None = None) -> dict:
    template = build_review_queue(db_path, limit)
    out_dir = S.ROOT / "data" / "ground_truth"
    out_dir.mkdir(parents=True, exist_ok=True)
    queue_path = out_dir / "review_queue.csv"
    df = template
    if review_path:
        same_as_queue = Path(review_path).resolve() == queue_path.resolve()
        reviews = pd.read_csv(review_path) if same_as_queue else review_path
        df = merge_reviews(template, reviews)
        if same_as_queue:
            df.to_csv(queue_path, index=False)
        else:
            template.to_csv(queue_path, index=False)
    else:
        template.to_csv(queue_path, index=False)
    validation = validate_review_queue(df)

    verified_path = verified_output or (S.ROOT / "data" / "ground_truth" / "verified.csv")
    export_available = validation["valid"] and (validation["verified_vulnerable"] or validation["verified_safe"])
    if export_available:
        export_verified(df, verified_path)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "database": str(db_path), "candidate_rows": int(len(df)),
        "candidate_vulnerable": int((df["candidate_type"] == "changed_function").sum()),
        "candidate_safe": int((df["candidate_type"] == "unchanged_function_in_fix_file").sum()),
        "validation": validation,
        "review_queue": str(queue_path), "verified_output": str(verified_path) if export_available else None,
        "important": "CVEfixes associations are candidates only. Verified labels require independent review evidence.",
    }
    S.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (S.REPORT_DIR / "ground_truth_report.json").write_text(json.dumps(report, indent=2))

    lines = ["# Ground-truth review report", "",
             "CVEfixes changed/unchanged associations are review candidates, not automatically verified ground truth.", "",
             f"- Candidate vulnerable rows: {report['candidate_vulnerable']:,}",
             f"- Candidate safe rows: {report['candidate_safe']:,}",
             f"- Verified vulnerable: {validation['verified_vulnerable']:,}",
             f"- Verified safe: {validation['verified_safe']:,}",
             f"- Not reviewed: {validation['not_reviewed']:,}", "",
             "A row can enter verified.csv only with reviewer, evidence, a stated review scope and method, and known limitations.",
             f"\nQueue: `{queue_path}`", ""]
    (S.REPORT_DIR / "ground_truth_report.md").write_text("\n".join(lines))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reviews", type=Path, default=None)
    parser.add_argument("--verified-output", type=Path, default=None)
    args = parser.parse_args()
    report = run(limit=args.limit, review_path=args.reviews, verified_output=args.verified_output)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
