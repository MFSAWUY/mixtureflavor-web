"""FastAPI application for FlavorGNN inference."""

from __future__ import annotations

import os
import time

import torch
from fastapi import FastAPI, HTTPException

from flavorgnn.api.schemas import (
    HealthResponse,
    LabelInfo,
    LabelsResponse,
    MoleculePrediction,
    PredictRequest,
    PredictResponse,
)
from flavorgnn.constants import FLAVOR_LABELS
from flavorgnn.inference import FlavorPredictor


MODEL_VERSION = "0.1.0"
CONFIG_PATH = os.getenv("FLAVORGNN_CONFIG", "configs/base.yaml")
CHECKPOINT_PATH = os.getenv("FLAVORGNN_CHECKPOINT", "")

app = FastAPI(title="FlavorGNN API", version=MODEL_VERSION)
predictor = FlavorPredictor(config_path=CONFIG_PATH, checkpoint_path=CHECKPOINT_PATH or None)


@app.get("/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_version=MODEL_VERSION,
        gpu_available=torch.cuda.is_available(),
    )


@app.get("/v1/flavor-labels", response_model=LabelsResponse)
def flavor_labels() -> LabelsResponse:
    labels = [LabelInfo(id=f"F{idx + 1:02d}", label=label) for idx, label in enumerate(FLAVOR_LABELS)]
    return LabelsResponse(labels=labels)


@app.post("/v1/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if req.smiles is None and req.inchi is None:
        raise HTTPException(status_code=400, detail="Either smiles or inchi must be provided.")
    if req.inchi is not None and req.smiles is None:
        raise HTTPException(status_code=501, detail="InChI conversion is not implemented in this demo.")

    smiles_list = [req.smiles] if isinstance(req.smiles, str) else list(req.smiles or [])
    if not smiles_list:
        raise HTTPException(status_code=400, detail="No valid SMILES input.")

    start = time.perf_counter()
    pred = predictor.predict(
        smiles=smiles_list,
        top_k=req.top_k,
        threshold=req.threshold,
        include_attention=req.include_attention,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return PredictResponse(
        predictions=[MoleculePrediction(**item) for item in pred],
        model_version=MODEL_VERSION,
        inference_ms=round(elapsed_ms, 3),
    )

