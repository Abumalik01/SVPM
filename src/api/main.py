"""Backend / API layer (FastAPI). No model logic here: it validates input, calls the ML layer, writes the log.

Run:  python -m uvicorn src.api.main:app --port 8000      then open http://127.0.0.1:8000/docs
Admin routes need the environment variable SVPM_ADMIN_TOKEN and the header X-Admin-Token.
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src import paths
from src.extract import lizard_extractor as ex
from src.ml import predict, registry
from src.ml.features import FEATURE_NAMES
from src.storage import log_db


class MetricsInput(BaseModel):
    nloc: int = Field(ge=0, le=1_000_000, description="Lines of code without comments or blanks (Lizard NLOC)")
    complexity: int = Field(ge=1, le=1_000_000, description="Cyclomatic complexity (Lizard CCN)")
    token_count: int = Field(ge=0, le=10_000_000, description="Number of tokens (Lizard)")
    n_params: int = Field(ge=0, le=1000, description="Number of parameters")
    top_nesting_level: int | None = Field(default=None, ge=0, le=1000, description="Only needed by models trained with it "
                                                                                   "(counted on the whole file, so a method inside a class counts those levels)")


def create_app(models_dir: Path | None = None, log_path: Path | None = None) -> FastAPI:
    models_dir = Path(models_dir or paths.MODELS_DIR)
    log_path = Path(log_path or paths.LOG_DB)
    predictor = predict.Predictor(models_dir)
    app = FastAPI(title="Software Vulnerability Prediction Prototype",
                  description="Decision-support estimate for C/C++ functions. " + predict.NOTICE)

    def _results(mode: str, items: list[dict]) -> dict:
        try:
            preds = predictor.predict_rows([{f: it.get(f) for f in FEATURE_NAMES} for it in items])
        except registry.RegistryError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        con = log_db.connect(log_path)
        try:
            for it, pr in zip(items, preds):
                log_db.log_prediction(con, mode, it.get("name"), it, pr)
        finally:
            con.close()
        rows = [{"function_name": it.get("name"), "start_line": it.get("start_line"),
                 "features": {f: it.get(f) for f in FEATURE_NAMES}, **pr} for it, pr in zip(items, preds)]
        rows.sort(key=lambda r: r["model_score"], reverse=True)
        for i, r in enumerate(rows, 1):
            r["rank_in_input"] = i
        return {"model_version": preds[0]["model_version"], "notice": predict.NOTICE, "results": rows}

    @app.get("/health")
    def health():
        return {"status": "ok", "active_model": predictor.active_info()}

    @app.post("/predict/metrics")
    def predict_metrics(m: MetricsInput):
        return _results("metrics", [m.model_dump()])

    @app.post("/predict/source")
    async def predict_source(file: UploadFile = File(...)):
        raw = await file.read(ex.MAX_BYTES + 1)
        try:
            source = ex.decode_source(file.filename or "", raw)
            funcs = ex.extract_functions(file.filename, source)
        except ex.ExtractionError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return _results("source", funcs)

    def admin(x_admin_token: str | None = Header(default=None)):
        expected = os.environ.get("SVPM_ADMIN_TOKEN")
        if not expected:
            raise HTTPException(status_code=503, detail="Admin interface is disabled: SVPM_ADMIN_TOKEN is not set.")
        if not x_admin_token or not secrets.compare_digest(x_admin_token, expected):
            raise HTTPException(status_code=401, detail="Invalid admin token.")

    @app.get("/admin/log", dependencies=[Depends(admin)])
    def admin_log(limit: int = 50):
        con = log_db.connect(log_path)
        try:
            return {"rows": log_db.recent(con, min(max(limit, 1), 500))}
        finally:
            con.close()

    @app.get("/admin/models", dependencies=[Depends(admin)])
    def admin_models():
        return {"active": registry.active_version(models_dir), "versions": registry.list_versions(models_dir)}

    return app


app = create_app()
