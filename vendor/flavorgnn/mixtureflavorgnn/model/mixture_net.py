"""Full mixture flavor prediction model."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from mixtureflavorgnn.constants import FLAVOR_LABELS
from mixtureflavorgnn.model.mol_encoder import MolecularEncoder
from mixtureflavorgnn.model.set_transformer import SetTransformerEncoder


def _masked_logsumexp(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    masked = values.masked_fill(~mask.unsqueeze(-1), float("-inf"))
    return torch.logsumexp(masked, dim=dim)


class MixtureFlavorNet(nn.Module):
    """Hierarchical mixture model with a single-molecule backbone and set transformer."""

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
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        isab_inducing_points: int = 16,
        backbone_checkpoint: str | Path | None = None,
        freeze_single_head: bool = True,
        num_labels: int = len(FLAVOR_LABELS),
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.molecular_encoder = MolecularEncoder(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            global_feature_dim=global_feature_dim,
            d_model=d_model,
            d_mix=d_mix,
            num_layers=num_layers,
            dropout=dropout,
            hidden_1=hidden_1,
            hidden_2=hidden_2,
            attention_heads=attention_heads,
            flavorgnn_checkpoint=backbone_checkpoint,
            freeze_single_head=freeze_single_head,
        )
        self.interaction_encoder = SetTransformerEncoder(
            d_model=d_mix,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            inducing_points=isab_inducing_points,
            dropout=dropout,
        )
        self.interaction_head = nn.Sequential(
            nn.Linear(d_mix, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )
        self.interaction_scale = nn.Parameter(torch.tensor(1.0))

    def set_backbone_trainable(self, trainable: bool) -> None:
        self.molecular_encoder.set_backbone_trainable(trainable)

    def parameter_groups(self, learning_rate: float, weight_decay: float, backbone_lr_scale: float) -> list[dict[str, object]]:
        backbone_params = []
        head_params = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("molecular_encoder.backbone."):
                backbone_params.append(parameter)
            else:
                head_params.append(parameter)

        groups = []
        if head_params:
            groups.append({"params": head_params, "lr": learning_rate, "weight_decay": weight_decay})
        if backbone_params:
            groups.append(
                {
                    "params": backbone_params,
                    "lr": learning_rate * backbone_lr_scale,
                    "weight_decay": weight_decay,
                }
            )
        return groups

    def _pack_components(
        self,
        component_embeddings: torch.Tensor,
        component_logits: torch.Tensor,
        component_owner: torch.Tensor,
        component_position: torch.Tensor,
        molecule_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_components = molecule_mask.shape
        d_mix = component_embeddings.shape[1]
        padded_embeddings = component_embeddings.new_zeros((batch_size, max_components, d_mix))
        padded_logits = component_logits.new_zeros((batch_size, max_components, self.num_labels))
        padded_embeddings[component_owner, component_position] = component_embeddings
        padded_logits[component_owner, component_position] = component_logits
        padded_embeddings = padded_embeddings * molecule_mask.unsqueeze(-1).to(component_embeddings.dtype)
        return padded_embeddings, padded_logits

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        return_interactions: bool = False,
        return_per_molecule_attribution: bool = False,
    ) -> dict[str, torch.Tensor]:
        encoded = self.molecular_encoder(batch)
        component_embeddings = encoded["mol_embedding"]
        component_logits = encoded["baseline_logits"]

        molecule_mask = batch["molecule_mask"]
        component_owner = batch["component_owner"]
        component_position = batch["component_position"]
        padded_embeddings, padded_logits = self._pack_components(
            component_embeddings=component_embeddings,
            component_logits=component_logits,
            component_owner=component_owner,
            component_position=component_position,
            molecule_mask=molecule_mask,
        )

        interaction_aware_embeddings, mixture_embedding, interaction_matrix = self.interaction_encoder(
            x=padded_embeddings,
            mask=molecule_mask,
            return_attention=return_interactions,
        )
        baseline_logit = _masked_logsumexp(padded_logits, mask=molecule_mask, dim=1)
        interaction_logit = self.interaction_head(mixture_embedding)
        interaction_scale = torch.clamp(self.interaction_scale, min=0.0, max=2.0)
        final_logit = baseline_logit + interaction_scale * interaction_logit

        out = {
            "logits": final_logit,
            "baseline_logit": baseline_logit,
            "interaction_logit": interaction_logit,
            "component_baseline_logits": padded_logits,
            "molecule_embeddings": padded_embeddings,
            "context_embeddings": interaction_aware_embeddings,
            "mixture_embedding": mixture_embedding,
            "node_attention": encoded["node_attention"],
        }
        if interaction_matrix is not None:
            out["interaction_matrix"] = interaction_matrix
        if return_per_molecule_attribution:
            flattened = interaction_aware_embeddings[molecule_mask]
            context_logits = self.molecular_encoder.score_context(flattened)
            padded_context_logits = padded_logits.new_zeros(padded_logits.shape)
            padded_context_logits[component_owner, component_position] = context_logits
            out["per_molecule_logits"] = padded_context_logits
        return out
