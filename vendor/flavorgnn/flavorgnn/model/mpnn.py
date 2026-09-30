"""Directed MPNN model for flavor prediction."""

from __future__ import annotations

import torch
from torch import nn

from flavorgnn.constants import FLAVOR_LABELS
from flavorgnn.model.layers import AttentionReadout, DirectedMPNNLayer


class FlavorDMPNN(nn.Module):
    """Directed MPNN with attention readout and multilabel head."""

    def __init__(
        self,
        node_feature_dim: int = 73,
        edge_feature_dim: int = 14,
        global_feature_dim: int = 16,
        d_model: int = 256,
        num_layers: int = 6,
        dropout: float = 0.3,
        hidden_1: int = 512,
        hidden_2: int = 256,
        attention_heads: int = 1,
        num_labels: int = len(FLAVOR_LABELS),
    ) -> None:
        super().__init__()

        self.num_labels = num_labels
        self.input_proj = nn.Sequential(
            nn.Linear(node_feature_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.layers = nn.ModuleList(
            [DirectedMPNNLayer(d_model=d_model, edge_feature_dim=edge_feature_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.readout = AttentionReadout(d_model=d_model, heads=attention_heads)

        self.classifier = nn.Sequential(
            nn.Linear(d_model + global_feature_dim, hidden_1),
            nn.BatchNorm1d(hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.BatchNorm1d(hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_2, num_labels),
        )

    def forward(self, batch: dict[str, torch.Tensor], return_attention: bool = False) -> dict[str, torch.Tensor]:
        x = batch["x"]
        edge_index = batch["edge_index"]
        edge_attr = batch["edge_attr"]
        global_features = batch["global_features"]
        batch_index = batch["batch"]
        num_graphs = int(global_features.shape[0])

        h0 = self.input_proj(x)
        h = h0
        for layer in self.layers:
            h = layer(h=h, h0=h0, edge_index=edge_index, edge_attr=edge_attr)

        graph_embedding, attention = self.readout(
            h=h,
            batch=batch_index,
            num_graphs=num_graphs,
            return_attention=return_attention,
        )
        fused = torch.cat([graph_embedding, global_features], dim=-1)
        logits = self.classifier(fused)

        out = {"logits": logits}
        if attention is not None:
            out["attention"] = attention
        return out

    @torch.inference_mode()
    def predict_proba(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval()
        logits = self.forward(batch=batch)["logits"]
        return torch.sigmoid(logits)

