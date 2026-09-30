"""Configuration helpers for mixture modeling."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flavorgnn.config import load_yaml_mapping


def _merge_dataclass(dc: Any, payload: dict[str, Any]) -> Any:
    for key, value in payload.items():
        if hasattr(dc, key):
            setattr(dc, key, value)
    return dc


@dataclass
class DataConfig:
    single_dataset_csv: str = "data/demo_flavor_dataset.csv"
    dataset_csv: str = "data/mixture_dataset.csv"
    split_dir: str = "data/mixture_splits"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42
    min_components: int = 2
    max_components: int = 5
    max_smiles_length: int = 2000
    use_rdkit_if_available: bool = True
    num_workers: int = 0
    mixture_count: int = 4096
    interaction_fraction: float = 0.65


@dataclass
class ModelConfig:
    d_model: int = 256
    d_mix: int = 256
    num_layers: int = 6
    dropout: float = 0.3
    hidden_1: int = 512
    hidden_2: int = 256
    attention_heads: int = 1
    transformer_layers: int = 4
    transformer_heads: int = 8
    isab_inducing_points: int = 16
    max_atomic_number: int = 43
    edge_feature_dim: int = 14
    global_feature_dim: int = 16
    max_mixture_size: int = 20


@dataclass
class TrainingConfig:
    batch_size: int = 64
    max_epochs: int = 40
    learning_rate: float = 1e-4
    backbone_lr_scale: float = 0.2
    weight_decay: float = 1e-5
    warmup_steps: int = 100
    label_smoothing: float = 0.05
    baseline_loss_weight: float = 0.5
    sparse_loss_weight: float = 0.01
    delta_supervision_weight: float = 0.25
    early_stopping_patience: int = 8
    grad_clip_norm: float = 5.0
    use_amp: bool = True
    freeze_backbone_steps: int = 200


@dataclass
class InferenceConfig:
    checkpoint_path: str = "artifacts/mixture/best_model.pt"
    top_k: int = 5
    threshold: float = 0.5
    include_per_molecule_attribution: bool = True
    include_interaction_matrix: bool = False


@dataclass
class ReportConfig:
    output_dir: str = "artifacts/mixture_report"
    calibration_bins: int = 10


@dataclass
class MixtureProjectConfig:
    experiment_name: str = "mixture_flavor_net"
    output_dir: str = "artifacts/mixture"
    device: str = "auto"
    seed: int = 42
    backbone_checkpoint: str = "artifacts/best_model.pt"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    report: ReportConfig = field(default_factory=ReportConfig)


def load_config(path: str | Path | None = None) -> MixtureProjectConfig:
    cfg = MixtureProjectConfig()
    if path is None:
        return cfg

    raw = load_yaml_mapping(path)
    top_level = {
        key: value
        for key, value in raw.items()
        if key not in {"data", "model", "training", "inference", "report"}
    }
    _merge_dataclass(cfg, top_level)

    if "data" in raw:
        _merge_dataclass(cfg.data, raw["data"] or {})
    if "model" in raw:
        _merge_dataclass(cfg.model, raw["model"] or {})
    if "training" in raw:
        _merge_dataclass(cfg.training, raw["training"] or {})
    if "inference" in raw:
        _merge_dataclass(cfg.inference, raw["inference"] or {})
    if "report" in raw:
        _merge_dataclass(cfg.report, raw["report"] or {})

    return cfg
