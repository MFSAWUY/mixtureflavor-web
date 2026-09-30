from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
import secrets
import threading
import time

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from flavorgnn.api.schemas import PredictRequest as LegacySingleRequest
from mixtureflavorgnn.api.schemas import MixturePredictRequest as LegacyMixtureRequest

from . import ROOT
from .backup import backup_database, start_backup_worker
from .chemistry import file_rows, text_rows, validate_rows
from .models import ModelService
from .schemas import InputProblem, PredictionRequest, PreviewRequest
from .settings import MAX_FILE_BYTES, NOTICE, STORAGE_NOTICE, Settings
from .storage import Store, utcnow

log = logging.getLogger(__name__)
security = HTTPBasic(auto_error=False)


class BodyLimitMiddleware:
    def __init__(self, app, limit=MAX_FILE_BYTES + 65536):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self.limit:
                error = InputProblem("body_too_large", "请求超过大小限制。", "Request exceeds the size limit.", status=413)
                return await JSONResponse({"error": error.payload()}, status_code=413)(scope, receive, send)
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        sent = False
        async def bounded_receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()
        await self.app(scope, bounded_receive, send)


def create_app(settings=None, service_factory=ModelService):
    settings = settings or Settings.from_env()
    @asynccontextmanager
    async def lifespan(app):
        settings.validate()
        app.state.store = Store(settings.database)
        app.state.models = service_factory(settings)
        app.state.gate = threading.BoundedSemaphore(2)
        app.state.auth_attempts = []
        app.state.auth_lock = threading.Lock()
        stop, thread = start_backup_worker(settings)
        app.state.backup_stop = stop
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=2)

    app = FastAPI(title="MixtureFlavor Research API", version="1.0.0", lifespan=lifespan,
                  description="Synthetic-benchmark demo. Prediction submissions are stored privately until administrator deletion. See /api/meta for the data notice.")
    app.add_middleware(BodyLimitMiddleware)
    routes = APIRouter()

    @app.exception_handler(InputProblem)
    async def input_error(request, exc):
        return JSONResponse({"error": exc.payload()}, status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def schema_error(request: Request, exc):
        issues = [{"location": list(e["loc"]), "message": e["msg"]} for e in exc.errors()]
        error = InputProblem("invalid_request", "提交格式无效，请检查输入和数据保存确认项。", "Invalid request. Check the input and storage acknowledgement.", issues)
        record_id = None
        if request.url.path.rstrip("/").endswith(("/predictions", "/v1/predict", "/v2/predict-mixture")):
            payload = exc.body if isinstance(exc.body, (dict, list, str)) else {"invalid_body": True}
            try:
                record_id = request.app.state.store.begin("invalid", payload, request.app.state.models.version)
                request.app.state.store.finish(record_id, error=error.payload())
            except Exception:
                log.exception("Could not record rejected prediction")
        return JSONResponse({"error": error.payload(), "record_id": record_id}, status_code=422)

    @app.middleware("http")
    async def response_headers(request, call_next):
        response = await call_next(request)
        if request.url.path.startswith(("/api", "/v1", "/v2")):
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Data-Policy"] = "prediction-submissions-retained-until-admin-deletion"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        return response

    def admin(request: Request, credentials: HTTPBasicCredentials | None = Depends(security)):
        now = time.monotonic()
        # Store only failure timestamps, not visitor IPs or passwords.
        with request.app.state.auth_lock:
            request.app.state.auth_attempts[:] = [t for t in request.app.state.auth_attempts if now - t < 60]
            if len(request.app.state.auth_attempts) >= 20:
                raise HTTPException(429, "Too many authentication failures. Retry in one minute.", headers={"Retry-After": "60"})
        valid = credentials is not None and secrets.compare_digest(credentials.username.encode(), settings.admin_user.encode()) and secrets.compare_digest(credentials.password.encode(), settings.admin_password.encode())
        if not valid:
            with request.app.state.auth_lock:
                request.app.state.auth_attempts.append(now)
            # Manual Authorization header is used by the UI; do not trigger a browser password dialog.
            raise HTTPException(401, "Administrator authentication required")
        return credentials.username

    def run_prediction(request, mode, rows, payload, include_attention=False):
        store, models = request.app.state.store, request.app.state.models
        record_id = store.begin(mode, payload, models.version)
        try:
            validated = validate_rows(rows, mode=mode, draw=False)
            if not request.app.state.gate.acquire(timeout=3):
                raise InputProblem("busy", "当前预测繁忙，请稍后重试。", "Prediction capacity is busy. Retry shortly.", status=503)
            try:
                result = models.predict(validated["molecules"], mode, include_attention)
            finally:
                request.app.state.gate.release()
            result.update(record_id=record_id, created_at=utcnow(), storage_notice=STORAGE_NOTICE, duplicates=validated["duplicates"])
            store.finish(record_id, result=result)
            return result
        except InputProblem as exc:
            store.finish(record_id, error=exc.payload())
            return JSONResponse({"error": exc.payload(), "record_id": record_id}, status_code=exc.status)
        except Exception:
            log.exception("Prediction failed; record_id=%s", record_id)
            error = InputProblem("prediction_failed", "预测未完成，请稍后重试或联系管理员并提供记录编号。", "Prediction failed. Retry or contact the administrator with the record ID.", status=500)
            try:
                store.finish(record_id, error=error.payload())
            except Exception:
                log.exception("Could not persist prediction failure; record_id=%s", record_id)
            return JSONResponse({"error": error.payload(), "record_id": record_id}, status_code=500)

    @routes.get("/health")
    @routes.get("/v1/health")
    @routes.get("/v2/health")
    def health(request: Request):
        with request.app.state.store.connect() as db:
            db.execute("SELECT 1")
        return {"status": "ok" if request.app.state.models.ready else "not_ready", "model_version": request.app.state.models.version, "device": "cpu", "gpu_available": False}

    @routes.get("/meta")
    def metadata(request: Request):
        return {**request.app.state.models.metadata(), "storage_notice": STORAGE_NOTICE,
                "limits": {"max_file_bytes": MAX_FILE_BYTES, "max_molecules": 100, "max_components": 5, "min_components": 2, "max_atoms_per_molecule": 128, "max_total_atoms": 4096},
                "backup_retention": "Latest 7 daily snapshots. Deleted records may remain in these snapshots until they rotate out."}

    @routes.get("/v1/flavor-labels")
    def labels(request: Request):
        return {"labels": [{"id": f"F{i:02d}", "label": item["id"]} for i, item in enumerate(request.app.state.models.metadata()["labels"], 1)]}

    @routes.post("/molecules/preview")
    def preview(body: PreviewRequest):
        return validate_rows(text_rows(body.text))

    @routes.post("/molecules/upload")
    async def upload(file: UploadFile = File(...)):
        try:
            content = await file.read(MAX_FILE_BYTES + 1)
            rows = await run_in_threadpool(file_rows, file.filename or "", content)
            return await run_in_threadpool(validate_rows, rows)
        finally:
            await file.close()

    @routes.post("/predictions")
    def prediction(body: PredictionRequest, request: Request):
        return run_prediction(request, body.mode, [m.model_dump() for m in body.molecules], body.model_dump())

    @routes.post("/v1/predict")
    def legacy_single(body: LegacySingleRequest, request: Request):
        if body.smiles is None and body.inchi is not None:
            record_id = request.app.state.store.begin("single", body.model_dump(), request.app.state.models.version)
            error = InputProblem("inchi_unsupported", "目前请使用 SMILES 输入。", "Use SMILES; InChI conversion is not implemented.", status=501)
            request.app.state.store.finish(record_id, error=error.payload())
            return JSONResponse({"error": error.payload(), "record_id": record_id}, status_code=501)
        raw = [body.smiles] if isinstance(body.smiles, str) else (body.smiles or [])
        result = run_prediction(request, "single" if len(raw) == 1 else "batch", [{"smiles": s} for s in raw], body.model_dump(), body.include_attention)
        if isinstance(result, JSONResponse):
            return result
        predictions = []
        for mol in result["molecules"]:
            ranked = sorted(mol["scores"], key=lambda s: s["score"], reverse=True)
            selected = [x for x in ranked if x["score"] >= body.threshold][:body.top_k] or ranked[:body.top_k]
            item = {"smiles": mol["smiles"], "flavors": [{"label": x["label"], "probability": x["score"]} for x in selected]}
            if body.include_attention:
                item["attention_weights"] = mol["attention_weights"]
            predictions.append(item)
        return {"predictions": predictions, "model_version": result["model_version"], "inference_ms": result["inference_ms"], "record_id": result["record_id"], "notice": NOTICE}

    @routes.post("/v2/predict-mixture")
    def legacy_mixture(body: LegacyMixtureRequest, request: Request):
        result = run_prediction(request, "mixture", [{"smiles": s} for s in body.mixture], body.model_dump())
        if isinstance(result, JSONResponse):
            return result
        mix = result["mixture"]
        indexed = list(enumerate(mix["scores"]))
        ranked = sorted(indexed, key=lambda x: x[1]["score"], reverse=True)
        selected = [(i, x) for j, (i, x) in enumerate(ranked) if j < body.top_k or x["score"] >= body.threshold]
        response = {"mixture_prediction": {"flavors": [{"label": x["label"], "probability": x["score"], "interaction_delta": mix["raw_interaction_logits"][i]} for i, x in selected]},
                    "model_version": result["model_version"], "inference_ms": result["inference_ms"], "record_id": result["record_id"], "notice": NOTICE,
                    "baseline_scores": mix["baseline_scores"], "score_deltas": mix["deltas"]}
        if body.include_per_molecule_attribution:
            response["per_molecule_attribution"] = [{"smiles": m["smiles"], "context_flavor_scores": {x["label"]: x["score"] for x in sorted(context, key=lambda x: x["score"], reverse=True)[:body.top_k]}} for m, context in zip(result["molecules"], mix["context_scores"])]
        if body.include_interaction_matrix:
            response["interaction_matrix"] = mix["attention_matrix"]
        return response

    @routes.get("/examples")
    def examples():
        return json.loads((ROOT / "examples" / "examples.json").read_text(encoding="utf-8"))

    @routes.get("/admin/records", dependencies=[Depends(admin)])
    def records(request: Request, limit: int = Query(25, ge=1, le=100), offset: int = Query(0, ge=0)):
        return request.app.state.store.list(limit, offset)

    @routes.get("/admin/export", dependencies=[Depends(admin)])
    def export(request: Request, format: str = Query("json", pattern="^(json|csv)$")):
        return StreamingResponse(request.app.state.store.export(format), media_type="application/json" if format == "json" else "text/csv",
                                 headers={"Content-Disposition": f'attachment; filename="mixtureflavor-records.{format}"'})

    @routes.get("/admin/records/{record_id}", dependencies=[Depends(admin)])
    def record(record_id: str, request: Request):
        item = request.app.state.store.get(record_id)
        if item is None:
            raise HTTPException(404, "Record not found")
        return item

    @routes.delete("/admin/records/{record_id}", dependencies=[Depends(admin)])
    def delete(record_id: str, request: Request):
        if not request.app.state.store.delete(record_id):
            raise HTTPException(404, "Record not found")
        return {"deleted": record_id, "backup_notice": "Old snapshots expire with the latest-7 backup rotation."}

    @routes.post("/admin/backup", dependencies=[Depends(admin)])
    def backup():
        path = backup_database(settings.database.resolve(), settings.backup_dir, settings.backup_keep)
        return {"backup": path.name}

    app.include_router(routes, prefix="/api")
    app.include_router(routes, include_in_schema=False)
    frontend = ROOT / "frontend" / "dist"
    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="website")
    return app


app = create_app()
