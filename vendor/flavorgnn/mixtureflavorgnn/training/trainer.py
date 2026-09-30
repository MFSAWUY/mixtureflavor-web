"""Trainer for the mixture model."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from mixtureflavorgnn.config import MixtureProjectConfig
from mixtureflavorgnn.data.mixture_dataset import collate_mixture_batch
from mixtureflavorgnn.training.losses import compute_mixture_loss
from mixtureflavorgnn.training.metrics import compute_mixture_metrics


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    warmup = max(warmup_steps, 1)
    total = max(total_steps, warmup + 1)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / float(total - warmup)
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)))
        return max(float(cosine), 1e-3)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


class MixtureTrainer:
    """Training loop and evaluation for MixtureFlavorNet."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: MixtureProjectConfig,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.pos_weight = pos_weight

        device_name = config.device
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.model.to(self.device)

        parameter_groups = self.model.parameter_groups(
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            backbone_lr_scale=self.config.training.backbone_lr_scale,
        )
        self.optimizer = AdamW(parameter_groups)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.config.training.use_amp and self.device.type == "cuda")
        self.global_step = 0
        self.backbone_unfrozen = self.config.training.freeze_backbone_steps <= 0
        self.model.set_backbone_trainable(self.backbone_unfrozen)

        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _counterfactual_probabilities(
        self,
        batch: dict[str, Any],
        base_probabilities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        before: list[torch.Tensor] = []
        after: list[torch.Tensor] = []
        flavor_indices: list[int] = []
        expected_changes: list[float] = []

        for idx, mixture_smiles in enumerate(batch["mixture_smiles"]):
            component_index = int(batch["focus_component_index"][idx].item())
            flavor_index = int(batch["focus_flavor_index"][idx].item())
            expected_change = float(batch["focus_expected_change"][idx].item())
            if component_index < 0 or flavor_index < 0 or component_index >= len(mixture_smiles):
                continue

            reduced_graphs = list(batch["component_graphs_nested"][idx])
            reduced_labels = batch["single_mol_labels"][idx][batch["molecule_mask"][idx]].detach().cpu()
            del reduced_graphs[component_index]
            keep_mask = torch.ones(reduced_labels.shape[0], dtype=torch.bool)
            keep_mask[component_index] = False
            reduced_labels = reduced_labels[keep_mask]
            reduced_smiles = [smiles for j, smiles in enumerate(mixture_smiles) if j != component_index]
            if len(reduced_smiles) == 0:
                continue

            reduced_item = {
                "mixture_id": batch["mixture_id"][idx],
                "smiles_list": reduced_smiles,
                "component_graphs": reduced_graphs,
                "y": batch["y"][idx].detach().cpu(),
                "single_mol_labels": reduced_labels,
                "interaction_type": batch["interaction_type"][idx],
                "interaction_targets": batch["interaction_targets"][idx].detach().cpu(),
                "interaction_mask": batch["interaction_mask"][idx].detach().cpu(),
                "focus_component_index": -1,
                "focus_flavor_index": flavor_index,
                "focus_expected_change": expected_change,
                "notes": batch["notes"][idx],
            }
            reduced_batch = collate_mixture_batch([reduced_item])
            reduced_batch = _move_to_device(reduced_batch, self.device)
            reduced_out = self.model(
                reduced_batch,
                return_interactions=False,
                return_per_molecule_attribution=False,
            )
            before.append(base_probabilities[idx].detach().cpu())
            after.append(torch.sigmoid(reduced_out["logits"]).detach().cpu()[0])
            flavor_indices.append(flavor_index)
            expected_changes.append(expected_change)

        if not before:
            empty_probs = torch.zeros((0, base_probabilities.shape[1]), dtype=torch.float32)
            empty_idx = torch.zeros((0,), dtype=torch.long)
            empty_change = torch.zeros((0,), dtype=torch.float32)
            return empty_probs, empty_probs, empty_idx, empty_change

        return (
            torch.stack(before, dim=0),
            torch.stack(after, dim=0),
            torch.tensor(flavor_indices, dtype=torch.long),
            torch.tensor(expected_changes, dtype=torch.float32),
        )

    def _run_epoch(
        self,
        dataloader: DataLoader,
        train: bool,
        compute_counterfactual: bool = False,
    ) -> dict[str, float]:
        mode = "train" if train else "eval"
        self.model.train(mode=train)

        total_loss = 0.0
        loss_terms = {"mixture_loss": 0.0, "baseline_loss": 0.0, "sparse_loss": 0.0, "delta_supervision_loss": 0.0}
        all_logits = []
        all_targets = []
        all_interaction_logits = []
        all_interaction_targets = []
        all_interaction_mask = []
        all_max_single_component_probs = []
        counterfactual_before = []
        counterfactual_after = []
        counterfactual_flavor_indices = []
        counterfactual_expected_changes = []

        for batch in dataloader:
            batch["component_graphs_nested"] = batch["component_graphs_nested"]
            batch = _move_to_device(batch, self.device)
            if (not self.backbone_unfrozen) and self.global_step >= self.config.training.freeze_backbone_steps:
                self.model.set_backbone_trainable(True)
                self.backbone_unfrozen = True

            with torch.set_grad_enabled(train):
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.config.training.use_amp and self.device.type == "cuda",
                ):
                    outputs = self.model(
                        batch,
                        return_interactions=False,
                        return_per_molecule_attribution=False,
                    )
                    pos_weight = self.pos_weight.to(self.device) if self.pos_weight is not None else None
                    loss, loss_dict = compute_mixture_loss(
                        outputs=outputs,
                        batch=batch,
                        pos_weight=pos_weight,
                        label_smoothing=self.config.training.label_smoothing,
                        baseline_loss_weight=self.config.training.baseline_loss_weight,
                        sparse_loss_weight=self.config.training.sparse_loss_weight,
                        delta_supervision_weight=self.config.training.delta_supervision_weight,
                    )

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.global_step += 1
                    if self.scheduler is not None:
                        self.scheduler.step()

            total_loss += float(loss.detach().cpu().item())
            for key in loss_terms:
                loss_terms[key] += float(loss_dict[key].detach().cpu().item())

            all_logits.append(outputs["logits"].detach().cpu())
            all_targets.append(batch["y"].detach().cpu())
            all_interaction_logits.append(outputs["interaction_logit"].detach().cpu())
            all_interaction_targets.append(batch["interaction_targets"].detach().cpu())
            all_interaction_mask.append(batch["interaction_mask"].detach().cpu())

            component_probs = torch.sigmoid(outputs["component_baseline_logits"].detach())
            masked_component_probs = component_probs.masked_fill(~batch["molecule_mask"].unsqueeze(-1), 0.0)
            all_max_single_component_probs.append(masked_component_probs.max(dim=1).values.detach().cpu())

            if compute_counterfactual and not train:
                base_probs = torch.sigmoid(outputs["logits"].detach())
                before, after, flavor_idx, expected = self._counterfactual_probabilities(batch, base_probs)
                if before.numel() > 0:
                    counterfactual_before.append(before)
                    counterfactual_after.append(after)
                    counterfactual_flavor_indices.append(flavor_idx)
                    counterfactual_expected_changes.append(expected)

        logits_tensor = torch.cat(all_logits, dim=0)
        targets_tensor = torch.cat(all_targets, dim=0)
        interaction_logits_tensor = torch.cat(all_interaction_logits, dim=0)
        interaction_targets_tensor = torch.cat(all_interaction_targets, dim=0)
        interaction_mask_tensor = torch.cat(all_interaction_mask, dim=0)
        max_single_component_probs_tensor = torch.cat(all_max_single_component_probs, dim=0)

        if counterfactual_before:
            before_tensor = torch.cat(counterfactual_before, dim=0)
            after_tensor = torch.cat(counterfactual_after, dim=0)
            flavor_idx_tensor = torch.cat(counterfactual_flavor_indices, dim=0)
            expected_tensor = torch.cat(counterfactual_expected_changes, dim=0)
        else:
            before_tensor = after_tensor = flavor_idx_tensor = expected_tensor = None

        metrics = compute_mixture_metrics(
            logits=logits_tensor,
            targets=targets_tensor,
            interaction_logits=interaction_logits_tensor,
            interaction_targets=interaction_targets_tensor,
            interaction_mask=interaction_mask_tensor,
            max_single_component_probabilities=max_single_component_probs_tensor,
            counterfactual_before_probabilities=before_tensor,
            counterfactual_after_probabilities=after_tensor,
            counterfactual_flavor_indices=flavor_idx_tensor,
            counterfactual_expected_changes=expected_tensor,
        )
        metrics[f"{mode}_loss"] = total_loss / max(len(dataloader), 1)
        for key, value in loss_terms.items():
            metrics[f"{mode}_{key}"] = value / max(len(dataloader), 1)
        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict[str, Any]:
        total_steps = max(len(train_loader) * self.config.training.max_epochs, 1)
        self.scheduler = _build_warmup_cosine_scheduler(
            optimizer=self.optimizer,
            warmup_steps=self.config.training.warmup_steps,
            total_steps=total_steps,
        )

        best_val_auc = -1.0
        best_epoch = -1
        epochs_without_improvement = 0
        history: list[dict[str, float]] = []

        for epoch in range(1, self.config.training.max_epochs + 1):
            train_metrics = self._run_epoch(train_loader, train=True, compute_counterfactual=False)
            val_metrics = self._run_epoch(val_loader, train=False, compute_counterfactual=False)

            merged = {"epoch": float(epoch)}
            merged.update(train_metrics)
            merged.update({f"val_{k}": v for k, v in val_metrics.items() if not k.startswith("eval_")})
            merged["val_loss"] = val_metrics["eval_loss"]
            history.append(merged)

            val_auc = val_metrics["macro_auroc"]
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                epochs_without_improvement = 0
                self._save_checkpoint(epoch=epoch, name="best_model.pt")
            else:
                epochs_without_improvement += 1

            if epoch % 5 == 0:
                self._save_checkpoint(epoch=epoch, name=f"checkpoint_epoch_{epoch}.pt")

            if epochs_without_improvement >= self.config.training.early_stopping_patience:
                break

        summary = {
            "best_val_macro_auroc": best_val_auc,
            "best_epoch": best_epoch,
            "history": history,
        }
        (self.output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, compute_counterfactual: bool = True) -> dict[str, float]:
        return self._run_epoch(dataloader, train=False, compute_counterfactual=compute_counterfactual)

    def _save_checkpoint(self, epoch: int, name: str) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "config": asdict(self.config),
        }
        torch.save(checkpoint, self.output_dir / name)
