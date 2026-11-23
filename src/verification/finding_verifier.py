"""Run reproducible C/C++ before/after verification cases.

This module does not turn a code change or sanitizer finding into a proven CVE
without a recorded reproducer and independent evidence. It produces auditable
before/after evidence for manual or automated review.

Run: python -m src.verification.finding_verifier --cases data/verification/cases.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from src.data import settings as S

SANITIZER_MARKERS = (
    "AddressSanitizer", "UndefinedBehaviorSanitizer", "runtime error:",
    "heap-buffer-overflow", "stack-buffer-overflow", "SEGV", "access violation",
)


def find_compiler(language: str = "C") -> str | None:
    names = ("g++", "clang++") if language.upper() == "C++" else ("gcc", "clang")
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def has_sanitizer_finding(output: str) -> bool:
    text = output.lower()
    return any(marker.lower() in text for marker in SANITIZER_MARKERS)


def compile_and_run(source: Path, language: str, arguments: list[str], timeout: int,
                    stdin: str = "") -> dict:
    compiler = find_compiler(language)
    if not compiler:
        return {"status": "compiler_unavailable", "output": "", "finding": False}

    suffix = ".cpp" if language.upper() == "C++" else ".c"
    executable = source.parent / ("verify.exe" if shutil.which("gcc") and ":/" in source.resolve().as_posix() else "verify")
    command = [compiler, "-x", "c++" if language.upper() == "C++" else "c", str(source),
               "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g", "-o", str(executable)]
    try:
        build = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except (subprocess.SubprocessError, OSError) as exc:
        return {"status": "build_error", "output": str(exc), "finding": False}
    if build.returncode != 0:
        output = (build.stdout + "\n" + build.stderr).strip()
        if "-lasan" in output or "-lubsan" in output:
            return {"status": "sanitizer_unavailable", "output": output, "finding": False}
        return {"status": "build_error", "output": output, "finding": False}

    try:
        run = subprocess.run([str(executable), *arguments], capture_output=True, text=True,
                             input=stdin, timeout=timeout, check=False)
        output = (run.stdout + "\n" + run.stderr).strip()
        finding = has_sanitizer_finding(output)
        return {"status": "executed", "exit_code": run.returncode, "output": output, "finding": finding}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "output": "execution exceeded timeout", "finding": False}
    except (subprocess.SubprocessError, OSError) as exc:
        return {"status": "execution_error", "output": str(exc), "finding": False}


def verify_case(case: dict, timeout: int = 30) -> dict:
    """Compile/run before and after source for one reproducible finding case."""
    required = {"candidate_id", "language", "source_before", "source_after"}
    missing = sorted(required - set(case))
    if missing:
        return {"candidate_id": case.get("candidate_id", ""), "status": "invalid_case",
                "errors": ["missing fields: " + ", ".join(missing)]}

    language = str(case["language"])
    if language not in {"C", "C++"}:
        return {"candidate_id": case["candidate_id"], "status": "invalid_case",
                "errors": ["language must be C or C++"]}

    arguments = [str(x) for x in case.get("arguments", [])]
    stdin = str(case.get("stdin", ""))
    source_before = str(case["source_before"])
    source_after = str(case["source_after"])
    with tempfile.TemporaryDirectory(prefix="svpm-finding-") as td:
        root = Path(td)
        before = root / "before.c"
        after = root / "after.c"
        before.write_text(source_before, encoding="utf-8")
        after.write_text(source_after, encoding="utf-8")
        before_result = compile_and_run(before, language, arguments, timeout, stdin)
        after_result = compile_and_run(after, language, arguments, timeout, stdin)

    if before_result["status"] != "executed" or after_result["status"] != "executed":
        status = "execution_not_observed"
    elif before_result["finding"] and not after_result["finding"]:
        status = "finding_removed_after_fix"
    elif before_result["finding"] and after_result["finding"]:
        status = "finding_persists_after_fix"
    elif not before_result["finding"] and after_result["finding"]:
        status = "unexpected_regression_after_fix"
    else:
        status = "no_sanitizer_finding_observed"

    return {
        "candidate_id": case["candidate_id"], "finding_type": case.get("finding_type", "unknown"),
        "status": status, "before": before_result, "after": after_result,
        "reproducer": case.get("reproducer", ""), "reviewer": case.get("reviewer", ""),
        "evidence_url": case.get("evidence_url", ""),
        "repo_url": case.get("repo_url", ""),
        "pre_fix_commit": case.get("pre_fix_commit", ""),
        "fixed_commit": case.get("fixed_commit", ""),
        "source_before_sha256": hashlib.sha256(source_before.encode("utf-8")).hexdigest(),
        "source_after_sha256": hashlib.sha256(source_after.encode("utf-8")).hexdigest(),
        "arguments": arguments, "stdin": stdin,
    }


def load_cases(path: Path) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else data["cases"]
    if path.suffix.lower() == ".csv":
        import pandas as pd
        return pd.read_csv(path).to_dict(orient="records")
    raise ValueError("cases must be a JSON or CSV file")


def run(cases_path: Path, timeout: int = 30) -> dict:
    cases = load_cases(cases_path)
    results = [verify_case(case, timeout) for case in cases]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(cases_path), "case_count": len(results),
        "summary": {status: sum(r["status"] == status for r in results)
                    for status in sorted({r["status"] for r in results})},
        "results": results,
        "important": "Sanitizer observations are reproducible evidence for a specific case, not proof of all related vulnerabilities.",
    }
    out_dir = S.REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "finding_verification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Finding verification report", "",
             "Sanitizer and before/after execution evidence for reproducible C/C++ cases.", "",
             f"- Cases: {len(results)}", ""]
    for status, count in report["summary"].items():
        lines.append(f"- {status}: {count}")
    lines += ["", "No result establishes broader vulnerability existence or absence.", ""]
    (out_dir / "finding_verification_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    report = run(args.cases, args.timeout)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
