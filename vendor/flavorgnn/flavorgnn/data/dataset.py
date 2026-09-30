"""Dataset and dataloader utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from flavorgnn.constants import FLAVOR_LABELS, LABEL_TO_INDEX
from flavorgnn.data.graph_builder import SmilesGraphFeaturizer


def parse_labels(label_text: str) -> list[str]:
    raw = (label_text or "").replace(",", ";")
    labels = [s.strip().lower() for s in raw.split(";") if s.strip()]
    return [name for name in labels if name in LABEL_TO_INDEX]


def labels_to_multihot(labels: list[str]) -> torch.Tensor:
    y = torch.zeros(len(FLAVOR_LABELS), dtype=torch.float32)
    for label in labels:
        y[LABEL_TO_INDEX[label]] = 1.0
    return y


@dataclass
class GraphRecord:
    smiles: str
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    global_features: torch.Tensor
    y: torch.Tensor


class FlavorGraphDataset(Dataset[dict[str, Any]]):
    """Loads CSV molecules and converts them to graph samples."""

    def __init__(
        self,
        csv_path: str | Path,
        featurizer: SmilesGraphFeaturizer,
        max_smiles_length: int = 2000,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.featurizer = featurizer
        self.max_smiles_length = max_smiles_length
        self.records: list[GraphRecord] = []
        self.skipped: list[tuple[str, str]] = []
        self._build()

    def _build(self) -> None:
        frame = pd.read_csv(self.csv_path)
        if "smiles" not in frame.columns or "labels" not in frame.columns:
            raise ValueError("Dataset CSV must contain columns: smiles, labels")

        for _, row in frame.iterrows():
            smiles = str(row["smiles"]).strip()
            if len(smiles) > self.max_smiles_length:
                self.skipped.append((smiles, "too_long"))
                continue

            labels = parse_labels(str(row["labels"]))
            if not labels:
                self.skipped.append((smiles, "no_valid_labels"))
                continue

            try:
                graph = self.featurizer.featurize(smiles)
            except Exception as exc:
                self.skipped.append((smiles, f"featurization_error:{exc}"))
                continue

            x = torch.tensor(graph["x"], dtype=torch.float32)
            edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
            edge_attr = torch.tensor(graph["edge_attr"], dtype=torch.float32)
            global_features = torch.tensor(graph["global_features"], dtype=torch.float32)
            y = labels_to_multihot(labels)

            self.records.append(
                GraphRecord(
                    smiles=smiles,
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    global_features=global_features,
                    y=y,
                )
            )

        if not self.records:
            raise RuntimeError(
                f"No valid molecules loaded from {self.csv_path}. "
                f"Skipped={len(self.skipped)}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        return {
            "smiles": record.smiles,
            "x": record.x,
            "edge_index": record.edge_index,
            "edge_attr": record.edge_attr,
            "global_features": record.global_features,
            "y": record.y,
        }


def collate_graph_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    x_list = []
    edge_index_list = []
    edge_attr_list = []
    global_list = []
    y_list = []
    smiles_list = []
    batch_index = []

    node_offset = 0
    for graph_id, item in enumerate(batch):
        x = item["x"]
        edge_index = item["edge_index"].clone()
        edge_attr = item["edge_attr"]
        n_nodes = x.shape[0]

        edge_index = edge_index + node_offset
        node_offset += n_nodes

        x_list.append(x)
        edge_index_list.append(edge_index)
        edge_attr_list.append(edge_attr)
        global_list.append(item["global_features"])
        y_list.append(item["y"])
        smiles_list.append(item["smiles"])
        batch_index.append(torch.full((n_nodes,), graph_id, dtype=torch.long))

    return {
        "x": torch.cat(x_list, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1),
        "edge_attr": torch.cat(edge_attr_list, dim=0),
        "global_features": torch.stack(global_list, dim=0),
        "y": torch.stack(y_list, dim=0),
        "batch": torch.cat(batch_index, dim=0),
        "smiles": smiles_list,
    }

