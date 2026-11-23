"""Model registry: versioned model bundles on disk, one active version.

models/<version>/model.joblib   the fitted scikit-learn model (scaling inside the pipeline where needed)
models/<version>/metadata.json  feature order, training ranges, settings, evaluation status, notes
models/active.json              {"version": "..."}  pointer to the model used for prediction
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import joblib

from src.ml.features import FEATURE_NAMES, FEATURE_SETS


class RegistryError(RuntimeError):
    pass


def _next_version(root: Path) -> str:
    nums = [int(p.name[1:]) for p in root.glob("v[0-9][0-9][0-9]") if p.is_dir()]
    return f"v{(max(nums) + 1) if nums else 1:03d}"


def save_candidate(model, metadata: dict, root: Path, version: str | None = None) -> str:
    """Save a model as a CANDIDATE. It does not become active until promote() is called."""
    root.mkdir(parents=True, exist_ok=True)
    version = version or _next_version(root)
    d = root / version
    if d.exists():
        raise RegistryError(f"version {version} already exists")
    d.mkdir()
    names = list(metadata.get("feature_names", FEATURE_NAMES))
    if names not in FEATURE_SETS.values():
        raise RegistryError("unknown feature set; use a set defined in features.FEATURE_SETS")
    meta = {**metadata, "version": version, "feature_names": names,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    joblib.dump(model, d / "model.joblib")
    (d / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    return version


def list_versions(root: Path) -> list[dict]:
    out = []
    for p in sorted(root.glob("v[0-9][0-9][0-9]")):
        m = p / "metadata.json"
        if m.exists():
            meta = json.loads(m.read_text())
            out.append({"version": p.name, "created_utc": meta.get("created_utc"),
                        "model_name": meta.get("model_name"), "sealed_test_evaluated": meta.get("sealed_test_evaluated")})
    return out


def promote(version: str, root: Path) -> None:
    if not (root / version / "model.joblib").exists():
        raise RegistryError(f"version {version} does not exist")
    (root / "active.json").write_text(json.dumps({"version": version}))


def active_version(root: Path) -> str | None:
    f = root / "active.json"
    return json.loads(f.read_text())["version"] if f.exists() else None


def load_version(version: str, root: Path) -> dict:
    d = root / version
    if not (d / "model.joblib").exists():
        raise RegistryError(f"version {version} does not exist")
    meta = json.loads((d / "metadata.json").read_text())
    if meta.get("feature_names") not in FEATURE_SETS.values():
        raise RegistryError("the model was trained with an unknown feature set; it cannot be used")
    return {"model": joblib.load(d / "model.joblib"), "metadata": meta}


def load_active(root: Path) -> dict:
    v = active_version(root)
    if v is None:
        raise RegistryError("no active model; train one and promote it first")
    return load_version(v, root)
