"""High-level inference interface for mixture prediction."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from mixtureflavorgnn.config import MixtureProjectConfig, load_config
from mixtureflavorgnn.constants import FLAVOR_LABELS
from mixtureflavorgnn.data.mixture_dataset import ComponentGraph, collate_mixture_batch
from mixtureflavorgnn.model import MixtureFlavorNet
from flavorgnn.data.graph_builder import SmilesGraphFeaturizer


class MixturePredictor:
    """Predict mixture flavors from lists of SMILES strings."""

    def __init__(
        self,
        config_path: str | Path = "configs/mixture_base.yaml",
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.config: MixtureProjectConfig = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.featurizer = SmilesGraphFeaturizer(
            use_rdkit_if_available=self.config.data.use_rdkit_if_available,
            max_atomic_number=self.config.model.max_atomic_number,
        )
        self.model: MixtureFlavorNet | None = None
        self.checkpoint_path = Path(checkpoint_path or self.config.inference.checkpoint_path)

    def _ensure_model(self, component_graphs: list[ComponentGraph]) -> None:
        if self.model is not None:
            return
        sample_graph = component_graphs[0]
        model = MixtureFlavorNet(
            node_feature_dim=sample_graph.x.shape[1],
            edge_feature_dim=sample_graph.edge_attr.shape[1],
            global_feature_dim=sample_graph.global_features.shape[0],
            d_model=self.config.model.d_model,
            d_mix=self.config.model.d_mix,
            num_layers=self.config.model.num_layers,
            dropout=self.config.model.dropout,
            hidden_1=self.config.model.hidden_1,
            hidden_2=self.config.model.hidden_2,
            attention_heads=self.config.model.attention_heads,
            transformer_layers=self.config.model.transformer_layers,
            transformer_heads=self.config.model.transformer_heads,
            isab_inducing_points=self.config.model.isab_inducing_points,
            backbone_checkpoint=self.config.backbone_checkpoint,
        )
        if self.checkpoint_path.exists():
            payload = torch.load(self.checkpoint_path, map_location=self.device)
            state_dict = payload.get("model_state", payload)
            model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        self.model = model

    def _build_single_item(self, smiles_list: list[str]) -> dict[str, Any]:
        component_graphs = []
        for smiles in smiles_list:
            graph = self.featurizer.featurize(smiles)
            component_graphs.append(
                ComponentGraph(
                    smiles=smiles,
                    x=torch.tensor(graph["x"], dtype=torch.float32),
                    edge_index=torch.tensor(graph["edge_index"], dtype=torch.long),
                    edge_attr=torch.tensor(graph["edge_attr"], dtype=torch.float32),
                    global_features=torch.tensor(graph["global_features"], dtype=torch.float32),
                )
            )

        return {
            "mixture_id": "PREDICT_0000",
            "smiles_list": smiles_list,
            "component_graphs": component_graphs,
            "y": torch.zeros(len(FLAVOR_LABELS), dtype=torch.float32),
            "single_mol_labels": torch.zeros((len(smiles_list), len(FLAVOR_LABELS)), dtype=torch.float32),
            "interaction_type": None,
            "interaction_targets": torch.zeros(len(FLAVOR_LABELS), dtype=torch.float32),
            "interaction_mask": torch.zeros(len(FLAVOR_LABELS), dtype=torch.bool),
            "focus_component_index": -1,
            "focus_flavor_index": -1,
            "focus_expected_change": 0.0,
            "notes": None,
        }

    @torch.inference_mode()
    def predict(
        self,
        smiles_list: list[str],
        top_k: int = 5,
        threshold: float = 0.5,
        include_per_molecule_attribution: bool = True,
        include_interaction_matrix: bool = False,
    ) -> dict[str, Any]:
        item = self._build_single_item(smiles_list)
        self._ensure_model(item["component_graphs"])
        assert self.model is not None

        batch = collate_mixture_batch([item])
        batch = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

        start = time.perf_counter()
        outputs = self.model(
            batch,
            return_interactions=include_interaction_matrix,
            return_per_molecule_attribution=include_per_molecule_attribution,
        )
        inference_ms = (time.perf_counter() - start) * 1000.0

        probs = torch.sigmoid(outputs["logits"]).detach().cpu()[0]
        interaction_delta = outputs["interaction_logit"].detach().cpu()[0]
        ranking = torch.argsort(probs, descending=True).tolist()

        flavors = []
        for idx in ranking:
            probability = float(probs[idx].item())
            if probability >= threshold or len(flavors) < top_k:
                flavors.append(
                    {
                        "label": FLAVOR_LABELS[idx],
                        "probability": probability,
                        "interaction_delta": float(interaction_delta[idx].item()),
                    }
                )
            if len(flavors) >= top_k and probability < threshold:
                break

        response: dict[str, Any] = {
            "mixture_prediction": {"flavors": flavors},
            "model_version": "2.0.0",
            "inference_ms": inference_ms,
        }

        if include_per_molecule_attribution and "per_molecule_logits" in outputs:
            per_molecule_probs = torch.sigmoid(outputs["per_molecule_logits"]).detach().cpu()[0]
            mask = batch["molecule_mask"].detach().cpu()[0]
            items = []
            for component_idx, smiles in enumerate(smiles_list):
                if not bool(mask[component_idx].item()):
                    continue
                row = per_molecule_probs[component_idx]
                top_indices = torch.argsort(row, descending=True)[:top_k].tolist()
                items.append(
                    {
                        "smiles": smiles,
                        "context_flavor_scores": {
                            FLAVOR_LABELS[idx]: float(row[idx].item()) for idx in top_indices
                        },
                    }
                )
            response["per_molecule_attribution"] = items

        if include_interaction_matrix and "interaction_matrix" in outputs:
            mask = batch["molecule_mask"].detach().cpu()[0]
            n_components = int(mask.sum().item())
            response["interaction_matrix"] = outputs["interaction_matrix"].detach().cpu()[0, :n_components, :n_components].tolist()

        return response
