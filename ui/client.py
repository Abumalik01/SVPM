"""Small HTTP client and table helpers for the Streamlit interface (no Streamlit code, so it is easy to test)."""
from __future__ import annotations

import os

import pandas as pd
import requests

DEFAULT_API = os.environ.get("SVPM_API_URL", "http://127.0.0.1:8000")
SHORT_NOTICE = ("This tool gives a statistical estimate to help decide where to review code first. It is not proof that a "
                "function is or is not vulnerable. It was trained on C and C++ functions and mostly reflects function size.")
UPLOAD_TYPES = ["c", "h", "cc", "cpp", "cxx", "hpp", "hh"]


class ApiError(RuntimeError):
    pass


def _check(resp: requests.Response) -> dict:
    if resp.ok:
        return resp.json()
    try:
        detail = resp.json().get("detail", resp.text)
    except ValueError:
        detail = resp.text
    if isinstance(detail, list):                       # FastAPI validation errors
        detail = "; ".join(f"{'.'.join(map(str, d.get('loc', [])[1:]))}: {d.get('msg')}" for d in detail)
    raise ApiError(str(detail))


def health(api: str) -> dict | None:
    try:
        r = requests.get(f"{api}/health", timeout=10)
        return r.json().get("active_model") if r.ok else None
    except requests.RequestException:
        raise ApiError(f"The prediction service is not reachable at {api}. Start it with: "
                       "python -m uvicorn src.api.main:app --port 8000")


def predict_metrics(api: str, values: dict) -> dict:
    return _check(requests.post(f"{api}/predict/metrics", json=values, timeout=30))


def predict_source(api: str, filename: str, content: bytes) -> dict:
    return _check(requests.post(f"{api}/predict/source", files={"file": (filename, content, "text/plain")}, timeout=120))


def admin_get(api: str, path: str, token: str) -> dict:
    return _check(requests.get(f"{api}{path}", headers={"X-Admin-Token": token}, timeout=30))


def results_frame(data: dict) -> pd.DataFrame:
    rows = []
    for r in data["results"]:
        rows.append({"Rank": r["rank_in_input"], "Function": r.get("function_name") or "(entered metrics)",
                     "Line": r.get("start_line"), "Model score": round(r["model_score"], 3),
                     "Percentile vs development functions": None if r.get("score_percentile") is None else round(r["score_percentile"]),
                     "Review priority": "Higher" if r["predicted_label"] == 1 else "Not flagged",
                     **{k: v for k, v in r["features"].items() if v is not None},
                     "Warnings": " | ".join(r["warnings"])})
    return pd.DataFrame(rows)
