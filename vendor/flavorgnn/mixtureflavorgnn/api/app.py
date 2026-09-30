"""FastAPI application for mixture prediction."""

from __future__ import annotations

from fastapi import FastAPI

from mixtureflavorgnn.api.schemas import MixturePredictRequest, MixturePredictionResponse
from mixtureflavorgnn.inference import MixturePredictor


app = FastAPI(title="MixtureFlavorNet API", version="2.0.0")
predictor = MixturePredictor()


@app.get("/v2/health")
def health() -> dict[str, object]:
    return {"status": "ok", "model_version": "2.0.0"}


@app.post("/v2/predict-mixture", response_model=MixturePredictionResponse)
def predict_mixture(request: MixturePredictRequest) -> MixturePredictionResponse:
    payload = predictor.predict(
        smiles_list=request.mixture,
        top_k=request.top_k,
        threshold=request.threshold,
        include_per_molecule_attribution=request.include_per_molecule_attribution,
        include_interaction_matrix=request.include_interaction_matrix,
    )
    return MixturePredictionResponse(**payload)
