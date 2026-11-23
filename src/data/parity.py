"""Phase D, step 2: feature-parity investigation (NO model training here).

Question: if a user supplies C/C++ code, will Lizard produce the same five
numbers that CVEfixes stored for training?

Two scenarios are tested on a seeded random sample of stored methods:
  WHOLE-FILE  The stored file text (code_before for pre-fix rows, code_after for
              post-fix rows) is analysed by the installed Lizard, the same way
              PyDriller did when CVEfixes was built. A mismatch here points to a
              Lizard VERSION difference.
  SNIPPET     Only the stored method text is analysed on its own, as when a user
              pastes one function. A mismatch here points to CONTEXT dependence
              (for example nesting inside a class or namespace).

Run:  python -m src.data.parity
Output: reports/parity_report.json and reports/parity_summary.md
"""
from __future__ import annotations

import ast
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

LANGUAGES = ("C", "C++")
FEATURES = ["nloc", "complexity", "token_count", "top_nesting_level", "n_params"]
SEED = 42
N_PER_CELL = 300            # rows per (language, label) cell
MAX_FILE_CHARS = 400_000    # skip very large files to keep the run fast
ADVISORY_EXACT_RATE = 0.95  # proposal U6, NOT a standard


# --------------------------------------------------------------------------
# Lizard helpers
# --------------------------------------------------------------------------
def lizard_functions(filename: str, source: str) -> list[dict]:
    import lizard
    info = lizard.analyze_file.analyze_source_code(filename, source)
    out = []
    for f in info.function_list:
        out.append({"name": f.name, "start_line": f.start_line, "end_line": f.end_line,
                    "nloc": f.nloc, "complexity": f.cyclomatic_complexity,
                    "token_count": f.token_count,
                    "top_nesting_level": f.top_nesting_level,
                    "n_params": f.parameter_count})
    return out


def stored_n_params(value) -> float:
    try:
        parsed = ast.literal_eval(str(value))
        return float(len(parsed)) if isinstance(parsed, (list, tuple)) else np.nan
    except (ValueError, SyntaxError):
        return np.nan


def find_match(funcs: list[dict], name: str, start_line, mode: str):
    """Pick the Lizard function that corresponds to the stored method."""
    same = [f for f in funcs if f["name"] == name]
    if mode == "whole_file":
        exact = [f for f in same if f["start_line"] == start_line]
        if exact:
            return exact[0]
        return same[0] if len(same) == 1 else None
    # snippet mode: a pasted method is normally the only function in the text
    if len(same) == 1:
        return same[0]
    return funcs[0] if len(funcs) == 1 else None


def compare(stored: dict, got: dict) -> dict:
    return {f: (float(stored[f]) == float(got[f]) if pd.notna(stored[f]) else False,
                (float(got[f]) - float(stored[f])) if pd.notna(stored[f]) else np.nan)
            for f in FEATURES}


# --------------------------------------------------------------------------
# sampling and execution
# --------------------------------------------------------------------------
def sample_rows(con: sqlite3.Connection, n_per_cell: int, seed: int) -> pd.DataFrame:
    light = pd.read_sql_query("""
        SELECT m.method_change_id, f.programming_language AS language, m.before_change
        FROM method_change m JOIN file_change f ON f.file_change_id = m.file_change_id
        WHERE f.programming_language IN ('C', 'C++')""", con)
    light["label"] = light.before_change.astype(str).str.strip().str.lower().map({"true": 1, "false": 0})
    light = light.dropna(subset=["label"])
    parts = [g.sample(n=min(n_per_cell, len(g)), random_state=seed)
             for _, g in light.groupby(["language", "label"])]
    ids = pd.concat(parts).method_change_id.tolist()
    frames = []
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        ph = ",".join("?" * len(chunk))
        frames.append(pd.read_sql_query(f"""
            SELECT m.method_change_id, m.file_change_id, m.name, m.start_line, m.code,
                   m.nloc, m.complexity, m.token_count, m.top_nesting_level, m.parameters,
                   m.before_change, f.filename, f.programming_language AS language,
                   f.code_before, f.code_after
            FROM method_change m JOIN file_change f ON f.file_change_id = m.file_change_id
            WHERE m.method_change_id IN ({ph})""", con, params=chunk))
    df = pd.concat(frames, ignore_index=True)
    df["label"] = df.before_change.astype(str).str.strip().str.lower().map({"true": 1, "false": 0})
    for c in ["nloc", "complexity", "token_count", "top_nesting_level", "start_line"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["n_params"] = df.parameters.map(stored_n_params)
    return df


def run_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (method, scenario, feature) with match info."""
    cache: dict = {}
    rows = []
    for r in df.itertuples(index=False):
        stored = {f: getattr(r, f) for f in FEATURES}
        for mode in ("whole_file", "snippet"):
            if mode == "whole_file":
                src = r.code_before if r.label == 1 else r.code_after
                key = (r.file_change_id, r.label)
            else:
                src, key = r.code, ("m", r.method_change_id)
            status, got = "ok", None
            if not isinstance(src, str) or not src.strip():
                status = "no_source"
            elif len(src) > MAX_FILE_CHARS:
                status = "too_large"
            else:
                if key not in cache:
                    try:
                        cache[key] = lizard_functions(r.filename, src)
                    except Exception:               # noqa: BLE001 - parser failure is a finding
                        cache[key] = None
                funcs = cache[key]
                if funcs is None:
                    status = "lizard_error"
                else:
                    got = find_match(funcs, r.name, r.start_line, mode)
                    if got is None:
                        status = "no_matching_function"
            if got is None:
                rows.append({"language": r.language, "label": r.label, "scenario": mode,
                             "status": status, "feature": None, "match": None, "diff": None,
                             "stored_nesting": stored["top_nesting_level"]})
                continue
            cmp_ = compare(stored, got)
            for f, (ok, diff) in cmp_.items():
                rows.append({"language": r.language, "label": r.label, "scenario": mode,
                             "status": "ok", "feature": f, "match": ok, "diff": diff,
                             "stored_nesting": stored["top_nesting_level"]})
    return pd.DataFrame(rows)


def summarise_parity(res: pd.DataFrame) -> dict:
    out = {"status_counts": {}, "feature_parity": []}
    for (lang, scen), g in res.groupby(["language", "scenario"]):
        status = g[g.feature.isna() | (g.feature == FEATURES[0])].status.value_counts()
        out["status_counts"][f"{lang}|{scen}"] = {k: int(v) for k, v in status.items()}
        ok = g[g.status == "ok"]
        for f in FEATURES:
            x = ok[ok.feature == f]
            if len(x):
                out["feature_parity"].append({
                    "language": lang, "scenario": scen, "feature": f, "n": int(len(x)),
                    "exact_match_rate": float(x.match.mean()),
                    "mean_abs_diff": float(x["diff"].abs().mean()),
                    "meets_advisory_threshold": bool(x.match.mean() >= ADVISORY_EXACT_RATE)})
    n = res[(res.feature == "top_nesting_level") & (res.status == "ok") & (res.scenario == "whole_file")]
    out["share_methods_with_nonzero_stored_nesting"] = (
        {lang: float((g.stored_nesting > 0).mean()) for lang, g in n.groupby("language")} if len(n) else {})
    return out


def render(rep: dict) -> str:
    L = ["# Feature parity report", "",
         f"Installed Lizard: {rep['lizard_version']}. CVEfixes requirements pin PyDriller ~=2.0; "
         "the Lizard version it used is not pinned (unknown).",
         f"Sample: up to {rep['n_per_cell']} rows per language and label; seed {rep['seed']}.", "",
         "## Match status", "", "| Language and scenario | Status counts |", "|---|---|"]
    L += [f"| {k} | {json.dumps(v)} |" for k, v in rep["status_counts"].items()]
    L += ["", "## Exact-match rate per feature", "",
          "| Language | Scenario | Feature | n | Exact match | Mean abs diff | >= 95% (advisory) |",
          "|---|---|---|---|---|---|---|"]
    for r in rep["feature_parity"]:
        L.append(f"| {r['language']} | {r['scenario']} | {r['feature']} | {r['n']} | "
                 f"{r['exact_match_rate']:.1%} | {r['mean_abs_diff']:.2f} | {r['meets_advisory_threshold']} |")
    L += ["", "WHOLE-FILE gaps suggest a Lizard version difference; SNIPPET gaps suggest context "
              "dependence. The 95% line is a proposal (U6) awaiting approval."]
    return "\n".join(L)


def run_parity(con: sqlite3.Connection, n_per_cell: int = N_PER_CELL, seed: int = SEED) -> dict:
    import lizard
    df = sample_rows(con, n_per_cell, seed)
    res = run_scenarios(df)
    rep = summarise_parity(res)
    rep.update({"lizard_version": lizard.version, "n_per_cell": n_per_cell, "seed": seed,
                "sampled_methods": int(len(df))})
    return rep


def main() -> int:
    from src.data import audit, settings as S
    con = audit.connect_ro(S.DB_PATH)
    try:
        rep = run_parity(con)
    finally:
        con.close()
    S.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (S.REPORT_DIR / "parity_report.json").write_text(json.dumps(rep, indent=2, default=str))
    (S.REPORT_DIR / "parity_summary.md").write_text(render(rep))
    print("Written: reports/parity_summary.md and parity_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
