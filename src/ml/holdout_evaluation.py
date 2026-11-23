"""Project-grouped evaluation against independently verified ground truth.

This module intentionally accepts only explicit, reviewed labels. CVEfixes
changed/unchanged labels are not loaded or inferred here.

Input CSV required columns:
  repo_url, ground_truth_label, model_score
Optional: model_predicted_class
Label 1 means independently verified vulnerable; 0 means independently verified safe.

Run: python -m src.ml.holdout_evaluation --input data/ground_truth/verified.csv
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import settings as S
from src.ml import evaluate

REQUIRED = {"repo_url", "ground_truth_label", "model_score"}


def prepare_holdout(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        raise ValueError("Independent holdout is missing columns: " + ", ".join(missing))
    if df["repo_url"].isna().any():
        raise ValueError("repo_url must be present for every independent holdout row")

    y = pd.to_numeric(df["ground_truth_label"], errors="coerce")
    s = pd.to_numeric(df["model_score"], errors="coerce")
    if y.isna().any() or not set(y.dropna().astype(int)).issubset({0, 1}):
        raise ValueError("ground_truth_label must contain only independently verified 0 or 1 labels")
    if s.isna().any():
        raise ValueError("model_score must be numeric")

    df = df.copy()
    df["y_true"] = y.astype(int)
    df["y_score"] = s.astype(float)
    if "model_predicted_class" in df:
        pred = pd.to_numeric(df["model_predicted_class"], errors="coerce")
        if pred.isna().any() or not set(pred.dropna().astype(int)).issubset({0, 1}):
            raise ValueError("model_predicted_class must contain only 0 or 1")
        df["y_pred"] = pred.astype(int)
    else:
        threshold = float(df["model_score_threshold"].dropna().iloc[0]) if "model_score_threshold" in df and df["model_score_threshold"].dropna().size else 0.0
        df["y_pred"] = (df["y_score"] >= threshold).astype(int)

    keep = ["repo_url", "y_true", "y_pred", "y_score"]
    if "language" in df:
        keep.append("language")
    return df[keep]


def evaluate_holdout(df: pd.DataFrame, n_boot: int = 1000, seed: int = 42) -> dict:
    if df.empty:
        raise ValueError("independent holdout is empty")
    if set(df.y_true) != {0, 1}:
        raise ValueError("independent holdout must contain both verified vulnerable and verified safe rows")

    boot = evaluate.bootstrap_ci_by_project(df, n_boot=n_boot, seed=seed)
    point = boot["point_estimate"]
    cm = np.asarray(point["confusion_matrix"], dtype=int)
    tn, fp, fn, tp = cm.ravel()

    per_language = {}
    if "language" in df:
        for language, group in df.groupby("language"):
            if set(group.y_true) == {0, 1}:
                metrics = evaluate.compute_metrics(group.y_true, group.y_pred, group.y_score)
                per_language[str(language)] = {"rows": int(len(group)), "projects": int(group.repo_url.nunique()),
                                               "precision": metrics["precision"], "recall": metrics["recall"],
                                               "f1": metrics["f1"]}

    return {
        "n_rows": int(len(df)), "n_projects": int(df.repo_url.nunique()),
        "n_verified_vulnerable": int(df.y_true.sum()), "n_verified_safe": int((df.y_true == 0).sum()),
        "point_estimate": point, "ci95": boot["ci95"], "n_boot": boot["n_boot"],
        "n_skipped_single_class": boot["n_skipped_single_class"],
        "seed": seed, "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "per_language": per_language,
        "interpretation": "These metrics apply only to the independently reviewed holdout. They do not prove exploitability or absence of vulnerability in unreviewed code.",
    }


def render(report: dict) -> str:
    ci = report["ci95"]
    def interval(key: str) -> str:
        if key not in ci:
            return "n/a"
        lo, hi = ci[key]
        return f"{report['point_estimate'][key]:.3f} ({lo:.3f} to {hi:.3f})"

    cm = report["confusion_matrix"]
    lines = ["# Independent holdout evaluation", "",
             "Ground truth consists only of independently reviewed and evidenced 0/1 labels.", "",
             f"- Rows: {report['n_rows']:,}", f"- Projects: {report['n_projects']:,}",
             f"- Verified vulnerable: {report['n_verified_vulnerable']:,}",
             f"- Verified safe: {report['n_verified_safe']:,}", "",
             "| Metric | Point estimate (95% project bootstrap CI) |", "|---|---|",
             f"| Precision | {interval('precision')} |", f"| Recall | {interval('recall')} |",
             f"| F1 | {interval('f1')} |", f"| ROC-AUC | {interval('roc_auc')} |",
             f"| Average precision | {interval('average_precision')} |", f"| Recall in top 10% | {interval('recall_top10pct')} |", "",
             f"Confusion matrix: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}.", "",
             "## Per-language point estimates", "",
             "| Language | Rows | Projects | Precision | Recall | F1 |", "|---|---|---|---|---|---|",
             *[f"| {language} | {values['rows']:,} | {values['projects']:,} | {values['precision']:.3f} | {values['recall']:.3f} | {values['f1']:.3f} |"
               for language, values in report.get("per_language", {}).items()], "",
             f"Project bootstrap: {report['n_boot']:,} resamples, seed {report['seed']}; "
             f"{report['n_skipped_single_class']} draws skipped with only one class.", "",
             report["interpretation"], ""]
    return "\n".join(lines)


def run(input_path: Path, n_boot: int = 1000, seed: int = 42) -> dict:
    df = prepare_holdout(input_path)
    report = evaluate_holdout(df, n_boot=n_boot, seed=seed)
    report.update({"generated_at": datetime.now(timezone.utc).isoformat(), "input": str(input_path)})
    S.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (S.REPORT_DIR / "independent_holdout_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (S.REPORT_DIR / "independent_holdout_evaluation.md").write_text(render(report), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.n_bootstrap, args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
