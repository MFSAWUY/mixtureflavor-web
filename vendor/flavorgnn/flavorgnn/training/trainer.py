"""Trainer implementation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from flavorgnn.config import ProjectConfig
from flavorgnn.training.losses import multilabel_bce_loss
from flavorgnn.training.metrics import compute_multilabel_metrics


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


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


class Trainer:
    """Handles training, evaluation, and checkpointing."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: ProjectConfig,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.pos_weight = pos_weight

        device_name = self.config.device
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.model.to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.config.training.use_amp and self.device.type == "cuda")
        self.global_step = 0

        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _run_epoch(self, dataloader: DataLoader, train: bool) -> dict[str, float]:
        mode = "train" if train else "eval"
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        all_logits = []
        all_targets = []

        for batch in dataloader:
            batch = _move_batch_to_device(batch, self.device)
            targets = batch["y"]

            with torch.set_grad_enabled(train):
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.config.training.use_amp and self.device.type == "cuda",
                ):
                    logits = self.model(batch)["logits"]
                    pos_weight = self.pos_weight.to(self.device) if self.pos_weight is not None else None
                    loss = multilabel_bce_loss(
                        logits=logits,
                        targets=targets,
                        pos_weight=pos_weight,
                        label_smoothing=self.config.training.label_smoothing,
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
            all_logits.append(logits.detach().cpu())
            all_targets.append(targets.detach().cpu())

        logits_tensor = torch.cat(all_logits, dim=0)
        targets_tensor = torch.cat(all_targets, dim=0)
        metrics = compute_multilabel_metrics(logits=logits_tensor, targets=targets_tensor)
        metrics[f"{mode}_loss"] = total_loss / max(len(dataloader), 1)
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
            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics = self._run_epoch(val_loader, train=False)

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
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        return self._run_epoch(dataloader, train=False)

    def _save_checkpoint(self, epoch: int, name: str) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "config": asdict(self.config),
        }
        torch.save(checkpoint, self.output_dir / name)

