"""API request/response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    smiles: str | list[str] | None = Field(default=None, description="SMILES string or list of SMILES strings")
    inchi: str | list[str] | None = Field(default=None, description="InChI input (placeholder in this demo)")
    top_k: int = Field(default=5, ge=1, le=24)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    include_attention: bool = False
    batch_size: int = Field(default=64, ge=1, le=10000)


class FlavorScore(BaseModel):
    label: str
    probability: float


class MoleculePrediction(BaseModel):
    smiles: str
    flavors: list[FlavorScore]
    attention_weights: list[float] | None = None


class PredictResponse(BaseModel):
    predictions: list[MoleculePrediction]
    model_version: str
    inference_ms: float


class HealthResponse(BaseModel):
    status: str
    model_version: str
    gpu_available: bool


class LabelInfo(BaseModel):
    id: str
    label: str


class LabelsResponse(BaseModel):
    labels: list[LabelInfo]

