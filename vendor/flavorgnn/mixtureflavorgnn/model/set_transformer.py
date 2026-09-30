"""Set-level interaction encoder for molecule embeddings."""

from __future__ import annotations

import torch
from torch import nn


def _apply_mask(tensor: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return tensor
    return tensor * mask.unsqueeze(-1).to(tensor.dtype)


class MultiheadAttentionBlock(nn.Module):
    """A residual multi-head attention block with feed-forward layers."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_mask: torch.Tensor | None = None,
        key_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_output, attn_weights = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=(~key_mask) if key_mask is not None else None,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        hidden = self.norm1(query + self.dropout(attn_output))
        hidden = _apply_mask(hidden, query_mask)
        ff = self.ffn(hidden)
        output = self.norm2(hidden + self.dropout(ff))
        output = _apply_mask(output, query_mask)
        return output, attn_weights if return_attention else None


class SetTransformerEncoder(nn.Module):
    """Permutation-invariant interaction encoder over molecule embeddings."""

    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        inducing_points: int = 16,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [MultiheadAttentionBlock(d_model=d_model, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.pool_seed = nn.Parameter(torch.randn(inducing_points, d_model) * 0.02)
        self.pool = MultiheadAttentionBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        hidden = _apply_mask(x, mask)
        attention_map = None

        for layer_index, layer in enumerate(self.layers):
            hidden, attn = layer(
                query=hidden,
                key_value=hidden,
                query_mask=mask,
                key_mask=mask,
                return_attention=return_attention and layer_index == len(self.layers) - 1,
            )
            if attn is not None:
                attention_map = attn.mean(dim=1)

        batch_size = hidden.shape[0]
        pool_seed = self.pool_seed.unsqueeze(0).expand(batch_size, -1, -1)
        pool_mask = torch.ones((batch_size, pool_seed.shape[1]), dtype=torch.bool, device=hidden.device)
        pooled, _ = self.pool(
            query=pool_seed,
            key_value=hidden,
            query_mask=pool_mask,
            key_mask=mask,
            return_attention=False,
        )
        mixture_embedding = pooled.mean(dim=1)
        return hidden, mixture_embedding, attention_map
