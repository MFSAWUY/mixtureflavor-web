"""Neural network layers for Directed MPNN."""

from __future__ import annotations

import torch
from torch import nn


def segment_softmax(scores: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Softmax over nodes per graph, supports [N] or [N, H] scores."""
    if scores.ndim == 1:
        out = torch.zeros_like(scores)
        for graph_id in range(num_graphs):
            mask = batch == graph_id
            if torch.any(mask):
                out[mask] = torch.softmax(scores[mask].float(), dim=0).to(out.dtype)
        return out

    out = torch.zeros_like(scores)
    for graph_id in range(num_graphs):
        mask = batch == graph_id
        if torch.any(mask):
            out[mask] = torch.softmax(scores[mask].float(), dim=0).to(out.dtype)
    return out


class DirectedMPNNLayer(nn.Module):
    """A single directed message passing layer."""

    def __init__(self, d_model: int, edge_feature_dim: int, dropout: float) -> None:
        super().__init__()
        self.message = nn.Linear(d_model + edge_feature_dim, d_model)
        self.update = nn.Linear(d_model * 2, d_model)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        h0: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        messages = self.message(torch.cat([h[src], edge_attr], dim=-1))
        messages = messages.to(h.dtype)
        aggregated = torch.zeros_like(h)
        aggregated.index_add_(0, dst, messages)
        updated = self.activation(self.update(torch.cat([h0, aggregated], dim=-1)))
        return self.dropout(updated)


class AttentionReadout(nn.Module):
    """Attention-weighted graph readout."""

    def __init__(self, d_model: int, heads: int = 1) -> None:
        super().__init__()
        self.heads = max(int(heads), 1)
        self.attn = nn.Linear(d_model, self.heads)

    def forward(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        scores = self.attn(h)  # [num_nodes, heads]
        alpha = segment_softmax(scores, batch, num_graphs)  # [num_nodes, heads]

        graph_embeddings = []
        for head_idx in range(self.heads):
            weighted = alpha[:, head_idx : head_idx + 1] * h
            pooled = torch.zeros((num_graphs, h.shape[1]), dtype=h.dtype, device=h.device)
            pooled.index_add_(0, batch, weighted)
            graph_embeddings.append(pooled)

        graph_emb = torch.stack(graph_embeddings, dim=1).mean(dim=1)

        if return_attention:
            attn_out = alpha.mean(dim=1)
            return graph_emb, attn_out
        return graph_emb, None

