"""Dataset split helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flavorgnn.constants import FLAVOR_LABELS, LABEL_TO_INDEX


def parse_labels(label_text: str) -> list[str]:
    raw = (label_text or "").replace(",", ";")
    labels = [s.strip().lower() for s in raw.split(";") if s.strip()]
    return [name for name in labels if name in LABEL_TO_INDEX]


def _labels_to_matrix(frame: pd.DataFrame) -> np.ndarray:
    y = np.zeros((len(frame), len(FLAVOR_LABELS)), dtype=np.int64)
    for i, label_text in enumerate(frame["labels"].fillna("").astype(str).tolist()):
        for name in parse_labels(label_text):
            y[i, LABEL_TO_INDEX[name]] = 1
    return y


def _random_split(
    n: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


def create_splits(
    dataset_csv: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> dict[str, Path]:
    """Create train/val/test splits and write them as CSV files."""

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    dataset_path = Path(dataset_csv)
    frame = pd.read_csv(dataset_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    y = _labels_to_matrix(frame)
    indices = np.arange(len(frame))

    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

        first_split = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=test_ratio, random_state=random_seed
        )
        train_val_idx, test_idx = next(first_split.split(indices, y))

        y_train_val = y[train_val_idx]
        val_fraction = val_ratio / (train_ratio + val_ratio)
        second_split = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=val_fraction, random_state=random_seed
        )
        train_rel_idx, val_rel_idx = next(second_split.split(train_val_idx, y_train_val))
        train_idx = train_val_idx[train_rel_idx]
        val_idx = train_val_idx[val_rel_idx]
    except Exception:
        train_idx, val_idx, test_idx = _random_split(
            n=len(frame),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=random_seed,
        )

    split_paths = {
        "train": out / "train.csv",
        "val": out / "val.csv",
        "test": out / "test.csv",
    }
    frame.iloc[train_idx].to_csv(split_paths["train"], index=False)
    frame.iloc[val_idx].to_csv(split_paths["val"], index=False)
    frame.iloc[test_idx].to_csv(split_paths["test"], index=False)
    return split_paths
