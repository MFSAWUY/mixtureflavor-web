"""Checkpoint load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_checkpoint(path: str | Path, map_location: torch.device | str | None = None) -> Any:
    """Load checkpoint with a safe default and backward-compatible fallback."""
    checkpoint_path = Path(path)
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError:
        # Older torch versions do not support weights_only.
        return torch.load(checkpoint_path, map_location=map_location)
    except RuntimeError as exc:
        # Some checkpoints may require full pickle loading.
        if "Weights only load failed" in str(exc):
            return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        raise
