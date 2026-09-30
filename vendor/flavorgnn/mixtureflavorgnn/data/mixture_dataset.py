"""Dataset and collation utilities for mixture learning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from flavorgnn.data.dataset import labels_to_multihot, parse_labels
from flavorgnn.data.graph_builder import SmilesGraphFeaturizer
from mixtureflavorgnn.constants import FLAVOR_LABELS, LABEL_TO_INDEX


def _parse_json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return [part.strip() for part in text.split("|") if part.strip()]


def _parse_json_dict(value: Any) -> dict[str, float]:
    if value is None:
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, float] = {}
    for key, raw_value in parsed.items():
        if key in LABEL_TO_INDEX:
            out[key] = float(raw_value)
    return out


@dataclass
class ComponentGraph:
    smiles: str
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    global_features: torch.Tensor


@dataclass
class MixtureRecord:
    mixture_id: str
    smiles_list: list[str]
    component_graphs: list[ComponentGraph]
    y: torch.Tensor
    single_mol_labels: torch.Tensor
    interaction_type: str | None
    interaction_targets: torch.Tensor
    interaction_mask: torch.Tensor
    focus_component_index: int
    focus_flavor_index: int
    focus_expected_change: float
    notes: str | None


class MixtureFlavorDataset(Dataset[dict[str, Any]]):
    """Loads mixture records from CSV and featurizes all component molecules."""

    def __init__(
        self,
        csv_path: str | Path,
        featurizer: SmilesGraphFeaturizer,
        max_smiles_length: int = 2000,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.featurizer = featurizer
        self.max_smiles_length = max_smiles_length
        self.records: list[MixtureRecord] = []
        self.skipped: list[tuple[str, str]] = []
        self._build()

    def _build(self) -> None:
        frame = pd.read_csv(self.csv_path)
        label_column = "mixture_labels" if "mixture_labels" in frame.columns else "labels"
        if "components" not in frame.columns or label_column not in frame.columns:
            raise ValueError("Mixture dataset CSV must contain columns: components and mixture_labels/labels")

        for _, row in frame.iterrows():
            mixture_id = str(row.get("mixture_id", ""))
            smiles_list = [str(smiles).strip() for smiles in _parse_json_list(row.get("components")) if str(smiles).strip()]
            if not smiles_list:
                self.skipped.append((mixture_id, "no_components"))
                continue
            if any(len(smiles) > self.max_smiles_length for smiles in smiles_list):
                self.skipped.append((mixture_id, "component_too_long"))
                continue

            labels = parse_labels(str(row.get(label_column, "")))
            if not labels:
                self.skipped.append((mixture_id, "no_valid_labels"))
                continue

            component_label_payload = _parse_json_list(row.get("component_labels"))
            component_label_tensors: list[torch.Tensor] = []
            for idx, component_labels in enumerate(component_label_payload):
                if isinstance(component_labels, list):
                    parsed_labels = [str(label).strip().lower() for label in component_labels if str(label).strip().lower() in LABEL_TO_INDEX]
                else:
                    parsed_labels = parse_labels(str(component_labels))
                if not parsed_labels and idx < len(smiles_list):
                    parsed_labels = []
                component_label_tensors.append(labels_to_multihot(parsed_labels))

            while len(component_label_tensors) < len(smiles_list):
                component_label_tensors.append(torch.zeros(len(FLAVOR_LABELS), dtype=torch.float32))

            try:
                graphs = []
                for smiles in smiles_list:
                    graph = self.featurizer.featurize(smiles)
                    graphs.append(
                        ComponentGraph(
                            smiles=smiles,
                            x=torch.tensor(graph["x"], dtype=torch.float32),
                            edge_index=torch.tensor(graph["edge_index"], dtype=torch.long),
                            edge_attr=torch.tensor(graph["edge_attr"], dtype=torch.float32),
                            global_features=torch.tensor(graph["global_features"], dtype=torch.float32),
                        )
                    )
            except Exception as exc:
                self.skipped.append((mixture_id, f"featurization_error:{exc}"))
                continue

            interaction_targets_dict = _parse_json_dict(row.get("interaction_targets"))
            interaction_targets = torch.zeros(len(FLAVOR_LABELS), dtype=torch.float32)
            interaction_mask = torch.zeros(len(FLAVOR_LABELS), dtype=torch.bool)
            for label, value in interaction_targets_dict.items():
                idx = LABEL_TO_INDEX[label]
                interaction_targets[idx] = float(value)
                interaction_mask[idx] = True

            focus_flavor = str(row.get("focus_flavor", "")).strip().lower()
            focus_flavor_index = LABEL_TO_INDEX.get(focus_flavor, -1)

            record = MixtureRecord(
                mixture_id=mixture_id,
                smiles_list=smiles_list,
                component_graphs=graphs,
                y=labels_to_multihot(labels),
                single_mol_labels=torch.stack(component_label_tensors, dim=0),
                interaction_type=str(row.get("interaction_type", "")).strip() or None,
                interaction_targets=interaction_targets,
                interaction_mask=interaction_mask,
                focus_component_index=int(row.get("focus_component_index", -1)),
                focus_flavor_index=focus_flavor_index,
                focus_expected_change=float(row.get("focus_expected_change", 0.0)),
                notes=str(row.get("notes", "")).strip() or None,
            )
            self.records.append(record)

        if not self.records:
            raise RuntimeError(f"No valid mixtures loaded from {self.csv_path}. Skipped={len(self.skipped)}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        return {
            "mixture_id": record.mixture_id,
            "smiles_list": record.smiles_list,
            "component_graphs": record.component_graphs,
            "y": record.y,
            "single_mol_labels": record.single_mol_labels,
            "interaction_type": record.interaction_type,
            "interaction_targets": record.interaction_targets,
            "interaction_mask": record.interaction_mask,
            "focus_component_index": record.focus_component_index,
            "focus_flavor_index": record.focus_flavor_index,
            "focus_expected_change": record.focus_expected_change,
            "notes": record.notes,
        }


def collate_mixture_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    x_list = []
    edge_index_list = []
    edge_attr_list = []
    global_feature_list = []
    node_batch = []
    component_owner = []
    component_position = []
    component_smiles: list[str] = []

    mixture_ids: list[str] = []
    mixture_smiles: list[list[str]] = []
    component_graphs_nested: list[list[ComponentGraph]] = []
    interaction_types: list[str | None] = []
    notes: list[str | None] = []

    batch_size = len(batch)
    num_labels = len(FLAVOR_LABELS)
    max_n = max(len(item["smiles_list"]) for item in batch)
    molecule_mask = torch.zeros((batch_size, max_n), dtype=torch.bool)
    y = torch.zeros((batch_size, num_labels), dtype=torch.float32)
    single_mol_labels = torch.zeros((batch_size, max_n, num_labels), dtype=torch.float32)
    interaction_targets = torch.zeros((batch_size, num_labels), dtype=torch.float32)
    interaction_mask = torch.zeros((batch_size, num_labels), dtype=torch.bool)
    focus_component_index = torch.full((batch_size,), -1, dtype=torch.long)
    focus_flavor_index = torch.full((batch_size,), -1, dtype=torch.long)
    focus_expected_change = torch.zeros((batch_size,), dtype=torch.float32)

    node_offset = 0
    graph_id = 0
    for mixture_idx, item in enumerate(batch):
        mixture_ids.append(item["mixture_id"])
        mixture_smiles.append(list(item["smiles_list"]))
        component_graphs_nested.append(list(item["component_graphs"]))
        interaction_types.append(item["interaction_type"])
        notes.append(item["notes"])
        y[mixture_idx] = item["y"]
        interaction_targets[mixture_idx] = item["interaction_targets"]
        interaction_mask[mixture_idx] = item["interaction_mask"]
        focus_component_index[mixture_idx] = int(item["focus_component_index"])
        focus_flavor_index[mixture_idx] = int(item["focus_flavor_index"])
        focus_expected_change[mixture_idx] = float(item["focus_expected_change"])

        for component_idx, graph in enumerate(item["component_graphs"]):
            n_nodes = graph.x.shape[0]
            x_list.append(graph.x)
            edge_index_list.append(graph.edge_index + node_offset)
            edge_attr_list.append(graph.edge_attr)
            global_feature_list.append(graph.global_features)
            node_batch.append(torch.full((n_nodes,), graph_id, dtype=torch.long))
            component_owner.append(mixture_idx)
            component_position.append(component_idx)
            component_smiles.append(graph.smiles)
            molecule_mask[mixture_idx, component_idx] = True
            single_mol_labels[mixture_idx, component_idx] = item["single_mol_labels"][component_idx]

            node_offset += n_nodes
            graph_id += 1

    return {
        "x": torch.cat(x_list, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1),
        "edge_attr": torch.cat(edge_attr_list, dim=0),
        "global_features": torch.stack(global_feature_list, dim=0),
        "batch": torch.cat(node_batch, dim=0),
        "component_owner": torch.tensor(component_owner, dtype=torch.long),
        "component_position": torch.tensor(component_position, dtype=torch.long),
        "component_smiles": component_smiles,
        "mixture_id": mixture_ids,
        "mixture_smiles": mixture_smiles,
        "component_graphs_nested": component_graphs_nested,
        "molecule_mask": molecule_mask,
        "y": y,
        "single_mol_labels": single_mol_labels,
        "interaction_targets": interaction_targets,
        "interaction_mask": interaction_mask,
        "interaction_type": interaction_types,
        "focus_component_index": focus_component_index,
        "focus_flavor_index": focus_flavor_index,
        "focus_expected_change": focus_expected_change,
        "notes": notes,
    }
