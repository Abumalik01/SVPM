"""Attach active-model scores to independently reviewed functions.

Run: python -m src.ml.score_ground_truth --input data/ground_truth/verified.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

from src import paths
from src.ml.predict import Predictor


def score_file(input_path: Path, output_path: Path,
               models_root: Path = paths.MODELS_DIR) -> dict:
    df = pd.read_csv(input_path)
    required = {"candidate_id", "repo_url", "ground_truth_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("verified input is missing columns: " + ", ".join(missing))
    if df.empty:
        raise ValueError("verified input has no rows")
    if df["candidate_id"].isna().any() or df["repo_url"].isna().any():
        raise ValueError("candidate_id and repo_url must be present for every row")
    labels = pd.to_numeric(df["ground_truth_label"], errors="coerce")
    if labels.isna().any() or not set(labels.unique()) <= {0, 1}:
        raise ValueError("ground_truth_label must contain only 0 or 1")

    predictor = Predictor(models_root)
    info = predictor.active_info()
    if info is None:
        raise ValueError("no active model is available")
    features = info["feature_names"]
    missing_features = sorted(set(features) - set(df.columns))
    if missing_features:
        raise ValueError("verified input is missing model features: " + ", ".join(missing_features))
    feature_values = df[features].apply(pd.to_numeric, errors="coerce")
    if feature_values.isna().any().any() or not np.isfinite(feature_values.to_numpy()).all():
        raise ValueError("model features must be finite numeric values")

    predictions = predictor.predict_rows(feature_values.to_dict(orient="records"))
    result = df.copy()
    result["model_score"] = [row["model_score"] for row in predictions]
    result["model_predicted_class"] = [row["predicted_label"] for row in predictions]
    result["model_threshold"] = [row["threshold"] for row in predictions]
    result["model_version"] = [row["model_version"] for row in predictions]
    result["model_warnings"] = [json.dumps(row["warnings"]) for row in predictions]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return {"input": str(input_path), "output": str(output_path), "rows": len(result),
            "projects": int(result["repo_url"].nunique()), "model_version": info["version"],
            "feature_names": features}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=paths.ROOT / "data/ground_truth/verified.csv")
    parser.add_argument("--output", type=Path,
                        default=paths.ROOT / "data/ground_truth/scored_verified.csv")
    parser.add_argument("--models-root", type=Path, default=paths.MODELS_DIR)
    args = parser.parse_args()
    print(json.dumps(score_file(args.input, args.output, args.models_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
