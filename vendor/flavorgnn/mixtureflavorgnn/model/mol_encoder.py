"""Wrap the single-molecule FlavorDMPNN backbone as an encoder."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from flavorgnn.model import FlavorDMPNN


class MolecularEncoder(nn.Module):
    """Single-molecule encoder reused by the mixture model."""

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        global_feature_dim: int,
        d_model: int = 256,
        d_mix: int = 256,
        num_layers: int = 6,
        dropout: float = 0.3,
        hidden_1: int = 512,
        hidden_2: int = 256,
        attention_heads: int = 1,
        flavorgnn_checkpoint: str | Path | None = None,
        freeze_single_head: bool = True,
    ) -> None:
        super().__init__()
        self.global_feature_dim = global_feature_dim
        self.fused_dim = d_model + global_feature_dim

        self.backbone = FlavorDMPNN(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            global_feature_dim=global_feature_dim,
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout,
            hidden_1=hidden_1,
            hidden_2=hidden_2,
            attention_heads=attention_heads,
        )

        self.mol_projection = nn.Sequential(
            nn.Linear(self.fused_dim, d_mix),
            nn.LayerNorm(d_mix),
        )
        self.context_projection = nn.Sequential(
            nn.Linear(d_mix, self.fused_dim),
            nn.LayerNorm(self.fused_dim),
        )

        if flavorgnn_checkpoint is not None and Path(flavorgnn_checkpoint).exists():
            payload = torch.load(flavorgnn_checkpoint, map_location="cpu")
            state_dict = payload.get("model_state", payload)
            self.backbone.load_state_dict(state_dict, strict=False)

        self.freeze_single_head = freeze_single_head
        if freeze_single_head:
            for parameter in self.backbone.classifier.parameters():
                parameter.requires_grad = False

    def set_backbone_trainable(self, trainable: bool) -> None:
        for name, parameter in self.backbone.named_parameters():
            if self.freeze_single_head and name.startswith("classifier."):
                parameter.requires_grad = False
            else:
                parameter.requires_grad = trainable

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = batch["x"]
        edge_index = batch["edge_index"]
        edge_attr = batch["edge_attr"]
        global_features = batch["global_features"]
        batch_index = batch["batch"]
        num_graphs = int(global_features.shape[0])

        h0 = self.backbone.input_proj(x)
        h = h0
        for layer in self.backbone.layers:
            h = layer(h=h, h0=h0, edge_index=edge_index, edge_attr=edge_attr)

        graph_embedding, attention = self.backbone.readout(
            h=h,
            batch=batch_index,
            num_graphs=num_graphs,
            return_attention=True,
        )
        fused = torch.cat([graph_embedding, global_features], dim=-1)
        baseline_logits = self.backbone.classifier(fused)
        mol_embedding = self.mol_projection(fused)
        return {
            "mol_embedding": mol_embedding,
            "baseline_logits": baseline_logits,
            "graph_embedding": graph_embedding,
            "fused_features": fused,
            "node_attention": attention,
        }

    def project_context(self, context_embeddings: torch.Tensor) -> torch.Tensor:
        return self.context_projection(context_embeddings)

    def score_context(self, context_embeddings: torch.Tensor) -> torch.Tensor:
        fused = self.project_context(context_embeddings)
        return self.backbone.classifier(fused)
