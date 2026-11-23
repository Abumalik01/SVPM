"""Independent evidence and holdout audit gate.

Step 5: use independent holdout evidence, repeat evaluation across projects and
languages, and record a separate audit trail. This gate refuses to certify a
model based on CVEfixes associations or missing reviewer evidence.

Run: python -m src.verification.independent_audit
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data import settings as S
from src.verification import exploitability

GROUND_TRUTH = S.ROOT / "data" / "ground_truth" / "verified.csv"
EXPLOITABILITY = S.ROOT / "data" / "verification" / "exploitability.csv"
FINDINGS = S.REPORT_DIR / "finding_verification_report.json"
HOLDOUT = GROUND_TRUTH


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def audit_verified_ground_truth(path: Path) -> dict:
    if not path.exists():
        return {"passed": False, "rows": 0, "reason": "verified ground-truth file is missing"}
    df = pd.read_csv(path)
    required = {"candidate_id", "ground_truth_status", "ground_truth_label", "reviewer",
                "evidence_url", "evidence_note", "reviewed_at", "repo_url"}
    missing = sorted(required - set(df.columns))
    if missing:
        return {"passed": False, "rows": int(len(df)), "missing_columns": missing}

    classes = set(df["ground_truth_label"].dropna().astype(int))
    evidence = (
        df["reviewer"].astype("string").fillna("").str.strip().ne("") &
        (
            df["evidence_url"].astype("string").fillna("").str.strip().ne("") |
            df["evidence_note"].astype("string").fillna("").str.strip().ne("")
        )
    )
    evidence_ok = bool(evidence.all())
    status_ok = set(df["ground_truth_status"]) <= {"vulnerable_verified", "safe_verified"}
    passed = len(df) > 0 and classes == {0, 1} and evidence_ok and status_ok
    return {
        "passed": bool(passed), "rows": int(len(df)), "projects": int(df["repo_url"].nunique()),
        "classes": sorted(int(v) for v in classes), "unique_reviewers": int(df["reviewer"].nunique()),
        "all_rows_have_evidence": evidence_ok, "statuses_valid": status_ok,
        "sha256": sha256(path),
    }


def audit_holdout_independence(verified_path: Path) -> dict:
    """Check that holdout projects do not occur in the CVEfixes development/test universe."""
    universe_path = S.ROOT / "data" / "processed" / "functions_capped.csv"
    if not verified_path.exists() or not universe_path.exists():
        return {"passed": False, "reason": "verified holdout or CVEfixes universe is unavailable"}

    holdout = pd.read_csv(verified_path)
    universe = pd.read_csv(universe_path, usecols=["repo_url"])
    holdout_projects = set(holdout["repo_url"].dropna().astype(str))
    training_projects = set(universe["repo_url"].dropna().astype(str))
    overlap = sorted(holdout_projects & training_projects)
    return {
        "passed": not overlap,
        "holdout_projects": len(holdout_projects),
        "overlap_projects": overlap,
        "reason": "holdout projects overlap CVEfixes project universe" if overlap else "no project overlap detected",
    }


def audit_findings(path: Path, verified_path: Path | None = None,
                   exploit_path: Path | None = None) -> dict:
    if not path.exists():
        return {"passed": False, "reason": "finding verification report is missing"}
    report = json.loads(path.read_text(encoding="utf-8"))
    results = report.get("results", [])
    invalid = sum(r.get("status") == "invalid_case" for r in results)
    executed = sum(r.get("status") not in {"execution_not_observed", "invalid_case"} for r in results)
    findings_removed = sum(r.get("status") == "finding_removed_after_fix" for r in results)
    passed = bool(results) and invalid == 0 and findings_removed > 0
    linked_candidates = None
    if verified_path is not None and exploit_path is not None:
        try:
            verified = pd.read_csv(verified_path)
            assessments = pd.read_csv(exploit_path)
            verified_ids = set(verified.loc[
                (verified.get("ground_truth_status") == "vulnerable_verified") &
                (pd.to_numeric(verified.get("ground_truth_label"), errors="coerce") == 1),
                "candidate_id"].astype(str))
            exploitable_ids = set(assessments.loc[
                assessments.get("assessment") == "exploitable_verified", "candidate_id"].astype(str))
            finding_ids = {str(r.get("candidate_id", "")) for r in results
                           if r.get("status") == "finding_removed_after_fix"}
            linked_candidates = sorted((verified_ids & exploitable_ids) & finding_ids)
            passed = passed and bool(linked_candidates)
        except (OSError, ValueError, KeyError, TypeError):
            linked_candidates = []
            passed = False
    return {
        "passed": passed, "cases": len(results), "invalid_cases": invalid,
        "executed": executed, "finding_removed_after_fix": findings_removed,
        "linked_verified_candidates": linked_candidates,
        "sha256": sha256(path),
    }


def audit_exploitability(path: Path) -> dict:
    if not path.exists():
        return {"passed": False, "reason": "exploitability assessment file is missing"}
    df = pd.read_csv(path)
    validation = exploitability.validate(df)
    summary = exploitability.summarise(df)
    return {"passed": bool(validation["valid"] and summary["exploitable_verified"] > 0),
            "validation": validation, "summary": summary, "sha256": sha256(path)}


def audit_holdout_metrics(path: Path) -> dict:
    """Require an actual independent holdout evaluation report, not just labels."""
    if not path.exists():
        return {"passed": False, "reason": "independent holdout evaluation report is missing"}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        point = report["point_estimate"]
        required = {"precision", "recall", "f1", "roc_auc", "average_precision", "recall_top10pct"}
        missing = sorted(required - set(point))
        passed = (report.get("n_rows", 0) > 0 and report.get("n_projects", 0) > 1
                  and report.get("n_verified_vulnerable", 0) > 0
                  and report.get("n_verified_safe", 0) > 0 and not missing)
        return {"passed": bool(passed), "rows": report.get("n_rows", 0),
                "projects": report.get("n_projects", 0), "missing_metrics": missing,
                "sha256": sha256(path)}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"passed": False, "reason": f"invalid independent holdout report: {exc}"}


def run() -> dict:
    ground = audit_verified_ground_truth(GROUND_TRUTH)
    independence = audit_holdout_independence(GROUND_TRUTH)
    findings = audit_findings(FINDINGS, GROUND_TRUTH, EXPLOITABILITY)
    exploit = audit_exploitability(EXPLOITABILITY)
    metrics = audit_holdout_metrics(S.REPORT_DIR / "independent_holdout_evaluation.json")

    checks = {
        "independent_verified_ground_truth": ground,
        "holdout_project_independence": independence,
        "reproducible_finding_evidence": findings,
        "exploitability_evidence_schema": exploit,
        "independent_holdout_metrics": metrics,
    }
    # Certification is intentionally strict. Missing evidence returns NOT_CERTIFIED.
    certified = all(check["passed"] for check in checks.values())
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "CERTIFIED" if certified else "NOT_CERTIFIED",
        "certified_for_vulnerability_existence": certified,
        "checks": checks,
        "important": "Certification requires independent evidence and project isolation. It does not prove that every unreviewed function is vulnerable or safe.",
    }

    S.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (S.REPORT_DIR / "independent_audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Independent evidence audit", "", f"Status: **{report['status']}**", "",]
    for name, check in checks.items():
        lines.append(f"- {name}: {'PASS' if check['passed'] else 'FAIL'}")
    lines += ["", report["important"], ""]
    (S.REPORT_DIR / "independent_audit_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    report = run()
    print(json.dumps(report, indent=2))
    return 0
