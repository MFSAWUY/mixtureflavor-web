"""API schemas for the mixture extension."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MixturePredictRequest(BaseModel):
    mixture: list[str] = Field(description="Unordered list of SMILES strings.")
    top_k: int = Field(default=5, ge=1, le=24)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    include_per_molecule_attribution: bool = True
    include_interaction_matrix: bool = False


class FlavorPrediction(BaseModel):
    label: str
    probability: float
    interaction_delta: float


class PerMoleculeAttribution(BaseModel):
    smiles: str
    context_flavor_scores: dict[str, float]


class MixturePredictionResponse(BaseModel):
    mixture_prediction: dict[str, list[FlavorPrediction]]
    per_molecule_attribution: list[PerMoleculeAttribution] | None = None
    interaction_matrix: list[list[float]] | None = None
    model_version: str
    inference_ms: float
