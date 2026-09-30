"""Configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised in user runtime
    yaml = None


@dataclass
class DataConfig:
    dataset_csv: str = "data/demo_flavor_dataset.csv"
    split_dir: str = "data/splits"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42
    min_label_count: int = 2
    use_rdkit_if_available: bool = True
    max_smiles_length: int = 2000
    num_workers: int = 0


@dataclass
class ModelConfig:
    d_model: int = 256
    num_layers: int = 6
    dropout: float = 0.3
    hidden_1: int = 512
    hidden_2: int = 256
    attention_heads: int = 1
    max_atomic_number: int = 43
    edge_feature_dim: int = 14
    global_feature_dim: int = 16


@dataclass
class TrainingConfig:
    batch_size: int = 64
    max_epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 100
    label_smoothing: float = 0.05
    aux_weight: float = 0.0
    early_stopping_patience: int = 10
    grad_clip_norm: float = 5.0
    use_amp: bool = False


@dataclass
class InferenceConfig:
    checkpoint_path: str = "artifacts/best_model.pt"
    top_k: int = 5
    threshold: float = 0.5
    return_attention: bool = False


@dataclass
class ProjectConfig:
    experiment_name: str = "flavorgnn_demo"
    output_dir: str = "artifacts"
    device: str = "auto"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


def _merge_dataclass(dc: Any, payload: dict[str, Any]) -> Any:
    for key, value in payload.items():
        if hasattr(dc, key):
            setattr(dc, key, value)
    return dc


def _coerce_scalar(value: str) -> Any:
    text = value.strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    try:
        if any(ch in text for ch in {".", "e", "E"}):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _simple_yaml_load(text: str) -> dict[str, Any]:
    """Parse the simple nested-mapping YAML used by this project."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = line.lstrip(" ")
        if ":" not in stripped:
            raise ValueError(f"Unsupported config line: {raw_line}")

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if not value:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _coerce_scalar(value)

    return root


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    raw_text = config_path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(raw_text) or {}
    return _simple_yaml_load(raw_text)


def load_config(path: str | Path | None = None) -> ProjectConfig:
    cfg = ProjectConfig()
    if path is None:
        return cfg

    raw = load_yaml_mapping(path)

    top_level = {k: v for k, v in raw.items() if k not in {"data", "model", "training", "inference"}}
    _merge_dataclass(cfg, top_level)

    if "data" in raw:
        _merge_dataclass(cfg.data, raw["data"] or {})
    if "model" in raw:
        _merge_dataclass(cfg.model, raw["model"] or {})
    if "training" in raw:
        _merge_dataclass(cfg.training, raw["training"] or {})
    if "inference" in raw:
        _merge_dataclass(cfg.inference, raw["inference"] or {})

    return cfg

