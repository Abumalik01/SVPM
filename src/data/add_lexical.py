"""Add code-pattern counts to the processed function dataset (no labels used, no model training).

Reads data/processed/functions_capped.csv, takes each function's text from the PRE-FIX file in the database
(file_change.code_before, lines start_line..end_line, exactly as build_universe did), and writes
data/processed/functions_capped_lex.csv with the same rows, the same function_id values and extra columns.
Integrity check: the md5 of the sliced text must equal the stored code_hash for every row; mismatches are reported.
The split assignment (by project) stays valid because rows do not change.

Run:  python -m src.data.add_lexical
Output: data/processed/functions_capped_lex.csv, reports/add_lexical_report.json
"""
from __future__ import annotations

import hashlib
import json
import sqlite3

import pandas as pd

from src.data.build_universe import function_text
from src.ml.lexical_features import LEX_NAMES, extract


def augment(df: pd.DataFrame, con: sqlite3.Connection):
    rows, mismatches, missing_files = [], 0, 0
    for fid, g in df.groupby("file_change_id", sort=False):
        r = con.execute("SELECT code_before FROM file_change WHERE file_change_id = ?", (int(fid),)).fetchone()
        if not r or not r[0]:
            missing_files += 1
            for i in g.index:
                rows.append((i, None, False))
            continue
        for i, f in g.iterrows():
            text = function_text(r[0], int(f.start_line), int(f.end_line))
            ok = hashlib.md5(text.encode("utf-8", "replace")).hexdigest() == f.code_hash
            mismatches += (not ok)
            rows.append((i, extract(text), ok))
    lex = pd.DataFrame([{**(d or {k: None for k in LEX_NAMES}), "text_hash_matches": ok} for _, d, ok in rows],
                       index=[i for i, _, _ in rows])
    out = df.join(lex)
    stats = {"rows": int(len(out)), "text_hash_mismatches": int(mismatches), "files_without_text": int(missing_files),
             "lexical_columns": LEX_NAMES}
    return out, stats


def run() -> dict:
    from src.data import audit, settings as S
    src = S.ROOT / "data" / "processed" / "functions_capped.csv"
    df = pd.read_csv(src)
    con = audit.connect_ro(S.DB_PATH)
    try:
        out, stats = augment(df, con)
    finally:
        con.close()
    if stats["text_hash_mismatches"] or stats["files_without_text"]:
        print("WARNING: integrity problems found; see the report before using the lexical features.")
    out.to_csv(S.ROOT / "data" / "processed" / "functions_capped_lex.csv", index=False)
    (S.REPORT_DIR / "add_lexical_report.json").write_text(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    s = run()
    print("Rows:", s["rows"], "| hash mismatches:", s["text_hash_mismatches"], "| files without text:", s["files_without_text"])
