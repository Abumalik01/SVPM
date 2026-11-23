"""Prediction service (ML layer, no HTTP code). The model bundle is loaded once and cached.

Output wording is deliberate: the score estimates resemblance to functions that past CVE-fixing commits
changed. It is not proof that a function is or is not vulnerable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.ml import pipeline, registry

NOTICE = ("This is a statistical estimate, not proof that a vulnerability exists or does not exist. "
          "The model compares five code metrics with functions that past CVE-fixing commits changed, "
          "using C and C++ data only. Larger functions tend to score higher. The score is a model score, "
          "not a calibrated probability; the percentile shows where it sits among the development functions "
          "the model was trained on. Use it only to help decide where to review first.")
CLASS_TEXT = {1: "Higher priority for review (resembles functions changed by CVE fixes)",
              0: "Not flagged (resembles functions not changed by CVE fixes)"}


class Predictor:
    def __init__(self, models_root: Path):
        self.root = Path(models_root)
        self._bundle = None
        self._version = None

    def _ensure(self) -> dict:
        v = registry.active_version(self.root)
        if v is None:
            raise registry.RegistryError("no active model; train one and promote it first")
        if self._bundle is None or v != self._version:        # reload only when the active version changes
            self._bundle, self._version = registry.load_version(v, self.root), v
        return self._bundle

    def active_info(self) -> dict | None:
        try:
            b = self._ensure()
        except registry.RegistryError:
            return None
        m = b["metadata"]
        return {"version": m["version"], "model_name": m.get("model_name"), "feature_names": m["feature_names"],
                "sealed_test_evaluated": m.get("sealed_test_evaluated", False), "note": m.get("note")}

    def predict_rows(self, rows: list[dict]) -> list[dict]:
        """rows: dicts with the five feature values. Returns one result per row, in the same order."""
        b = self._ensure()
        meta, model = b["metadata"], b["model"]
        names = meta["feature_names"]
        missing = sorted({f for r in rows for f in names if r.get(f) is None})
        if missing:
            raise ValueError(f"missing feature value(s) needed by the active model: {missing}")
        X = np.array([[float(r[f]) for f in names] for r in rows])
        scores = pipeline.score(model, X)
        thr = float(meta.get("decision_threshold", 0.5))
        ranges = meta.get("training_ranges", {})
        out = []
        for r, s in zip(rows, scores):
            warn = [f"{f} = {r[f]} is outside the range seen in training "
                    f"({ranges[f]['min']} to {ranges[f]['max']}); the estimate is less reliable"
                    for f in names if f in ranges and not (ranges[f]["min"] <= r[f] <= ranges[f]["max"])]
            label = int(s >= thr)
            q = meta.get("score_quantiles")
            pct = None if not q else float(np.interp(s, q, np.linspace(0, 100, len(q))))
            out.append({"model_score": float(s), "score_percentile": pct, "predicted_label": label, "predicted_class": CLASS_TEXT[label],
                        "threshold": thr, "model_version": meta["version"], "warnings": warn})
        return out
