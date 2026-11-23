"""Fit the selected model on ALL development data and save it as a registry CANDIDATE (prototype step).

* Uses development data only. The sealed test set is NOT read here; metadata records
  sealed_test_evaluated = false until the single final evaluation is done.
* The model is saved with training ranges, so the API can warn about out-of-range inputs.

Run:  python -m src.ml.final_fit [--model DecisionTree] [--promote]
Default model = the U9 selection stored in reports/cv_results_v2.json.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from src import paths
from src.ml import features, pipeline, registry


def build_bundle(dev, model_name: str, feature_names=None, log_transform: bool = False, notes: dict | None = None,
                 model=None):
    """Fit one model on ALL development data and return (model, metadata). No sealed-test data is used here.
    If an already fitted `model` (trained on exactly this development data) is passed, it is reused, not refitted."""
    names = list(feature_names or features.FEATURE_NAMES)
    X, y = dev[names].to_numpy(float), dev.label.to_numpy(int)
    if model is None:
        model = pipeline.make_model(model_name, y, log_transform=log_transform).fit(X, y)
    ranges = {f: {"min": float(dev[f].min()), "p99": float(np.percentile(dev[f], 99)), "max": float(dev[f].max())}
              for f in names}
    scores = pipeline.score(model, X)
    meta = {"model_name": model_name, "feature_names": names, "log_transform": log_transform,
            "hyperparameters": pipeline.HYPERPARAMETERS[model_name],
            "training_rows": int(len(dev)), "training_label_1": int((y == 1).sum()),
            "training_ranges": ranges,
            # SVM scores are decision-function values: its class boundary is 0, not 0.5
            "decision_threshold": 0.5 if hasattr(model, "predict_proba") else 0.0,
            "score_quantiles": [float(q) for q in np.percentile(scores, np.linspace(0, 100, 101))],
            "sealed_test_evaluated": False,
            "label_definition": "label 1 = function changed by a CVE-fixing commit; label 0 = other function in the same "
                                "pre-fix file (not verified vulnerable or safe)", **(notes or {})}
    return model, meta


def fit_and_save(dev, model_name: str, models_root, notes: dict, promote: bool = False,
                 feature_names=None, log_transform: bool = False) -> str:
    model, meta = build_bundle(dev, model_name, feature_names, log_transform, notes)
    version = registry.save_candidate(model, meta, models_root)
    if promote:
        registry.promote(version, models_root)
    return version


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--features", choices=sorted(features.FEATURE_SETS), default="v1_five")
    ap.add_argument("--log-transform", action="store_true")
    ap.add_argument("--cv-file", default=None, help="CV results file that holds the U9 selection")
    a = ap.parse_args()
    cvname = a.cv_file or ("cv_results_v2.json" if a.features == "v1_five" and not a.log_transform else "cv_results_v2_four_log.json")
    cv = json.loads((paths.REPORT_DIR / cvname).read_text())["primary"]
    name = a.model or cv["selection"]["selected"]
    names = features.FEATURE_SETS[a.features]
    dev = features.load_development_v2(paths.PROCESSED_DIR / "functions_capped.csv", paths.PROCESSED_DIR / "split_assignment_v2.csv", names)
    notes = {"note": "Development model: fitted on all development data. The sealed test set has not been used yet.",
             "cv_mean_f1": cv["models"][name]["summary"]["f1"]["mean"], "selection_trace": cv["selection"]}
    v = fit_and_save(dev, name, paths.MODELS_DIR, notes, promote=a.promote, feature_names=names, log_transform=a.log_transform)
    print(f"Saved {name} as {v}" + (" and made it active." if a.promote else " (candidate; not active)."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
