"""Subset augmentation helpers for mixtures."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable


def enumerate_nontrivial_subsets(items: list[str], min_size: int = 2) -> list[list[str]]:
    subsets: list[list[str]] = []
    for size in range(min_size, len(items)):
        for subset in combinations(items, size):
            subsets.append(list(subset))
    return subsets


def limited_subsets(items: list[str], limit: int = 8, min_size: int = 2) -> list[list[str]]:
    subsets = enumerate_nontrivial_subsets(items, min_size=min_size)
    return subsets[: max(limit, 0)]


def flatten_subsets(batch_subsets: Iterable[list[list[str]]]) -> list[list[str]]:
    flattened: list[list[str]] = []
    for subsets in batch_subsets:
        flattened.extend(subsets)
    return flattened
