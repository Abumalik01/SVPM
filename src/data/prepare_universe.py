"""Target v2, step 3: apply the pre-registered caps and write the processed function dataset
(NO model training here).

The caps are read from reports/universe_caps.json ("recommended"), i.e. exactly the option chosen by
the rule that was fixed before the grid was seen. The file contains metrics and identifiers only
(no source code).

Run:  python -m src.data.prepare_universe
Output: data/processed/functions_capped.csv, reports/prepare_universe_report.json and .md
"""
from __future__ import annotations

import hashlib
import json

import pandas as pd

from src.data import universe_caps as uc
from src.data.build_universe import FEATURES, summarise

KEEP = ["function_id", "file_change_id", "commit_hash", "repo_url", "filename", "language", "name",
        "start_line", "end_line", *FEATURES, "code_hash", "label", "n_changed_in_file"]


def apply_recommended(df: pd.DataFrame, rec: dict) -> pd.DataFrame:
    if rec is None:
        raise ValueError("no cap option met the criteria; widen the grid before preparing the dataset")
    return uc.apply_caps(df, rec["pos_commit_cap"], rec["pos_project_cap"], rec["neg_project_cap"])


def per_language(d: pd.DataFrame) -> dict:
    out = {}
    for lang, g in d.groupby("language"):
        out[lang] = {"functions": int(len(g)), "label_1": int((g.label == 1).sum()),
                     "positive_share": float((g.label == 1).mean()), "projects": int(g.repo_url.nunique())}
    return out


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def render(rep: dict) -> str:
    s, c = rep["final"], rep["caps_used"]
    pct = lambda x: f"{x:.1%}"
    L = ["# Processed function dataset (target v2)", "",
         "Label 1 = function changed by the CVE-fixing commit (pre-fix code); label 0 = other function in the same "
         "pre-fix file. Not independently verified vulnerable and safe functions.", "",
         f"Caps used (chosen by the rule fixed in advance): label-1 per commit {c['pos_commit_cap']}, label-1 per project "
         f"{c['pos_project_cap']}, label-0 per project {c['neg_project_cap']}; seed {rep['seed']}.", "",
         f"- functions: {s['functions']:,}; label 1: {s['label_1_changed']:,} ({pct(s['positive_share'])}); "
         f"label 0: {s['label_0_unchanged']:,}",
         f"- projects: {s['projects']:,}; files: {s['files']:,}; commits: {s['commits']:,}",
         f"- top project share: {pct(s['top_project_share'])} of functions, {pct(s['top_project_share_of_positives'])} of label 1",
         "", "NOTE: capping removes only label-0 functions from large projects, so the share of label 1 "
         f"({pct(s['positive_share'])}) is higher than in the uncleaned universe ({pct(rep['before']['positive_share'])}). "
         "Precision depends on this share; real software will have a lower share.", "",
         "| Language | Functions | Label 1 | Positive share | Projects |", "|---|---|---|---|---|"]
    L += [f"| {k} | {v['functions']:,} | {v['label_1']:,} | {pct(v['positive_share'])} | {v['projects']} |"
          for k, v in rep["per_language"].items()]
    L += ["", f"Input sha256 `{rep['input_sha256']}`", f"Output sha256 `{rep['output_sha256']}`", ""]
    return "\n".join(L)


def run() -> dict:
    from src.data import settings as S
    src = S.INTERIM_DIR / "function_universe.csv"
    caps = json.loads((S.REPORT_DIR / "universe_caps.json").read_text())
    rec = caps["recommended"]
    df = pd.read_csv(src)
    out = apply_recommended(df, rec)
    path = S.ROOT / "data" / "processed" / "functions_capped.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out[KEEP].to_csv(path, index=False)
    rep = {"seed": uc.SEED, "caps_used": {k: rec[k] for k in ("pos_commit_cap", "pos_project_cap", "neg_project_cap")},
           "criteria": caps["criteria"], "before": summarise(df), "final": summarise(out),
           "per_language": per_language(out), "input_sha256": sha256(src), "output_sha256": sha256(path)}
    (S.REPORT_DIR / "prepare_universe_report.json").write_text(json.dumps(rep, indent=2, default=str))
    (S.REPORT_DIR / "prepare_universe_summary.md").write_text(render(rep))
    return rep


if __name__ == "__main__":
    r = run()
    print("Functions:", r["final"]["functions"], "| label 1:", r["final"]["label_1_changed"],
          "| written: data/processed/functions_capped.csv")
