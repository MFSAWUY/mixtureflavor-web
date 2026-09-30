"""Prediction utility."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from flavorgnn.checkpoint import load_checkpoint
from flavorgnn.config import ProjectConfig, load_config
from flavorgnn.constants import FLAVOR_LABELS
from flavorgnn.data.dataset import collate_graph_batch
from flavorgnn.data.graph_builder import SmilesGraphFeaturizer
from flavorgnn.model import FlavorDMPNN


def _make_model(config: ProjectConfig, node_feature_dim: int = 73) -> FlavorDMPNN:
    return FlavorDMPNN(
        node_feature_dim=node_feature_dim,
        edge_feature_dim=config.model.edge_feature_dim,
        global_feature_dim=config.model.global_feature_dim,
        d_model=config.model.d_model,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        hidden_1=config.model.hidden_1,
        hidden_2=config.model.hidden_2,
        attention_heads=config.model.attention_heads,
        num_labels=len(FLAVOR_LABELS),
    )


class FlavorPredictor:
    """High-level predictor for single or batch SMILES strings."""

    def __init__(
        self,
        config_path: str | Path = "configs/base.yaml",
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.config = load_config(config_path)
        self.featurizer = SmilesGraphFeaturizer(
            use_rdkit_if_available=self.config.data.use_rdkit_if_available,
            max_atomic_number=self.config.model.max_atomic_number,
        )
        self.model = _make_model(self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        ckpt = Path(checkpoint_path or self.config.inference.checkpoint_path)
        if ckpt.exists():
            payload = load_checkpoint(ckpt, map_location=self.device)
            state_dict = payload.get("model_state", payload)
            self.model.load_state_dict(state_dict, strict=False)

    def _build_batch(self, smiles_list: list[str]) -> dict[str, Any]:
        items = []
        for smiles in smiles_list:
            graph = self.featurizer.featurize(smiles)
            item = {
                "smiles": smiles,
                "x": torch.tensor(graph["x"], dtype=torch.float32),
                "edge_index": torch.tensor(graph["edge_index"], dtype=torch.long),
                "edge_attr": torch.tensor(graph["edge_attr"], dtype=torch.float32),
                "global_features": torch.tensor(graph["global_features"], dtype=torch.float32),
                "y": torch.zeros(len(FLAVOR_LABELS), dtype=torch.float32),
            }
            items.append(item)
        return collate_graph_batch(items)

    @torch.inference_mode()
    def predict(
        self,
        smiles: str | list[str],
        top_k: int = 5,
        threshold: float = 0.5,
        include_attention: bool = False,
    ) -> list[dict[str, Any]]:
        smiles_list = [smiles] if isinstance(smiles, str) else list(smiles)
        batch = self._build_batch(smiles_list)
        for key, value in list(batch.items()):
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)

        out = self.model(batch, return_attention=include_attention)
        probs = torch.sigmoid(out["logits"]).detach().cpu()
        attention = out.get("attention")
        batch_index = batch["batch"].detach().cpu()
        if attention is not None:
            attention = attention.detach().cpu().tolist()

        results = []
        for i, smi in enumerate(smiles_list):
            row = probs[i]
            sorted_idx = torch.argsort(row, descending=True)
            selected = []
            for idx in sorted_idx.tolist():
                prob = float(row[idx].item())
                if prob >= threshold:
                    selected.append({"label": FLAVOR_LABELS[idx], "probability": prob})
                if len(selected) >= top_k:
                    break
            if not selected:
                for idx in sorted_idx[:top_k].tolist():
                    selected.append({"label": FLAVOR_LABELS[idx], "probability": float(row[idx].item())})

            item = {"smiles": smi, "flavors": selected}
            if include_attention and attention is not None:
                node_attention = [attention[node_idx] for node_idx, g in enumerate(batch_index.tolist()) if g == i]
                item["attention_weights"] = node_attention
            results.append(item)

        return results
