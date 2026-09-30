"""Read-only CPU inference with strict checkpoint and feature-contract checks."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from rdkit import rdBase

from flavorgnn.config import load_config as single_config
from flavorgnn.constants import FLAVOR_LABELS
from flavorgnn.data.graph_builder import _featurize_with_rdkit
from flavorgnn.data.dataset import collate_graph_batch
from flavorgnn.model import FlavorDMPNN
from mixtureflavorgnn.config import load_config as mixture_config
from mixtureflavorgnn.data.mixture_dataset import ComponentGraph, collate_mixture_batch
from mixtureflavorgnn.model import MixtureFlavorNet

from . import ROOT
from .settings import NOTICE, DATASET_KIND

LABELS_ZH = ["甜味", "苦味", "酸味", "咸味", "鲜味", "果香", "花香", "辛香", "草本", "脂香", "烟熏", "坚果", "焦糖", "香草", "柑橘", "薄荷", "木香", "泥土", "硫香", "酒香", "涩感", "清凉", "金属", "厚味"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scores(values):
    values = values.detach().cpu().tolist()
    if len(values) != 24 or not np.isfinite(values).all():
        raise RuntimeError("Model produced an invalid score vector")
    return [{"label": label, "score": float(value)} for label, value in zip(FLAVOR_LABELS, values)]


class ModelService:
    def __init__(self, settings):
        self.vendor = settings.vendor
        self.ready = False
        torch.set_num_threads(settings.threads)
        manifest = json.loads((self.vendor / "model-manifest.json").read_text(encoding="utf-8"))
        for relative, expected in manifest["sha256"].items():
            path = self.vendor / relative
            if not path.is_file() or sha256(path) != expected:
                raise RuntimeError(f"Missing or changed model asset: {relative}. Validate and regenerate the manifest before deploying a different model.")
        if rdBase.rdkitVersion != manifest["rdkit_version"]:
            raise RuntimeError(f"RDKit {manifest['rdkit_version']} is required by the feature contract; found {rdBase.rdkitVersion}.")
        sc = single_config(self.vendor / "configs/base.yaml")
        mc = mixture_config(self.vendor / "configs/mixture_base.yaml")
        if not sc.data.use_rdkit_if_available or not mc.data.use_rdkit_if_available:
            raise RuntimeError("The web model contract requires RDKit features.")
        if sc.model.max_atomic_number != mc.model.max_atomic_number:
            raise RuntimeError("Single and mixture feature configurations disagree.")
        self.atomic_max = sc.model.max_atomic_number
        feature = self._graph("CCO")
        dimensions = [feature["x"].shape[1], feature["edge_attr"].shape[1], feature["global_features"].shape[0]]
        if dimensions != manifest["feature_dimensions"] or list(FLAVOR_LABELS) != manifest["labels"]:
            raise RuntimeError("Feature dimensions or label order do not match the model manifest.")
        kwargs = asdict(sc.model)
        kwargs.pop("max_atomic_number")
        kwargs["node_feature_dim"] = dimensions[0]
        self.single_model = FlavorDMPNN(**kwargs)
        self.single_model.load_state_dict(self._state("artifacts/best_model.pt"), strict=True)
        self.single_model.eval()
        kwargs = asdict(mc.model)
        kwargs.pop("max_atomic_number")
        kwargs.pop("max_mixture_size")
        kwargs["node_feature_dim"] = dimensions[0]
        # The full mixture checkpoint contains the backbone. Never initialize a missing part silently.
        self.mixture_model = MixtureFlavorNet(**kwargs, backbone_checkpoint=None)
        self.mixture_model.load_state_dict(self._state("artifacts/mixture/best_model.pt"), strict=True)
        self.mixture_model.eval()
        self.version = "mf-web-1.0.0-" + manifest["sha256"]["artifacts/best_model.pt"][:8] + "-" + manifest["sha256"]["artifacts/mixture/best_model.pt"][:8]
        self.manifest = manifest
        self.metrics = {
            "single": json.loads((self.vendor / "artifacts/test_metrics.json").read_text()),
            "mixture": json.loads((self.vendor / "artifacts/mixture/test_metrics.json").read_text()),
        }
        warm = [{"id": "M001", "name": "Ethanol", "smiles": "CCO", "canonical_smiles": "CCO"}, {"id": "M002", "name": "Acetic acid", "smiles": "CC(=O)O", "canonical_smiles": "CC(=O)O"}]
        result = self.predict(warm, "mixture")
        reference_path = self.vendor / "reference-predictions.json"
        if not reference_path.is_file():
            raise RuntimeError("The reference prediction fixture is missing.")
        reference = json.loads(reference_path.read_text())
        actual = [x["score"] for x in result["mixture"]["scores"]]
        if not np.allclose(actual, reference["mixture_scores"], atol=2e-5, rtol=2e-5):
            raise RuntimeError("Warm-up prediction differs from the reference model. Check feature/runtime compatibility.")
        actual_single = [[x["score"] for x in m["scores"]] for m in result["molecules"]]
        if not np.allclose(actual_single, reference["single_scores"], atol=2e-5, rtol=2e-5):
            raise RuntimeError("Single-molecule warm-up differs from the reference model.")
        self.ready = True

    def _state(self, relative):
        payload = torch.load(self.vendor / relative, map_location="cpu", weights_only=True)
        return payload.get("model_state", payload)

    def _graph(self, smiles):
        features = _featurize_with_rdkit(smiles, self.atomic_max)
        if features is None:
            raise ValueError("RDKit could not featurize this molecule; fallback parsing is disabled.")
        item = {
            "smiles": smiles,
            "x": torch.tensor(features["x"], dtype=torch.float32),
            "edge_index": torch.tensor(features["edge_index"], dtype=torch.long).reshape(2, -1),
            "edge_attr": torch.tensor(features["edge_attr"], dtype=torch.float32).reshape(-1, 14),
            "global_features": torch.tensor(features["global_features"], dtype=torch.float32),
            "y": torch.zeros(24),
        }
        for key in ["x", "edge_attr", "global_features"]:
            if not torch.isfinite(item[key]).all():
                raise ValueError("Non-finite molecular features")
        return item

    @torch.inference_mode()
    def predict(self, molecules, mode, include_attention=False):
        start = time.perf_counter()
        graphs = [self._graph(m["smiles"]) for m in molecules]
        results = []
        for offset in range(0, len(graphs), 16):
            batch = collate_graph_batch(graphs[offset:offset + 16])
            output = self.single_model(batch, return_attention=include_attention)
            for j, row in enumerate(torch.sigmoid(output["logits"])):
                m = molecules[offset + j]
                result = {k: m[k] for k in ["id", "name", "smiles", "canonical_smiles"]}
                result["scores"] = scores(row)
                if include_attention:
                    result["attention_weights"] = output["attention"][batch["batch"] == j].flatten().tolist()
                results.append(result)
        response = {"mode": mode, "molecules": results, "mixture": None, "model_version": self.version, "dataset_kind": DATASET_KIND, "notice": NOTICE}
        if mode == "mixture":
            components = [ComponentGraph(**{k: g[k] for k in ["smiles", "x", "edge_index", "edge_attr", "global_features"]}) for g in graphs]
            item = {
                "mixture_id": "WEB", "smiles_list": [m["smiles"] for m in molecules], "component_graphs": components,
                "y": torch.zeros(24), "single_mol_labels": torch.zeros((len(molecules), 24)),
                "interaction_type": None, "interaction_targets": torch.zeros(24), "interaction_mask": torch.zeros(24, dtype=torch.bool),
                "focus_component_index": -1, "focus_flavor_index": -1, "focus_expected_change": 0.0, "notes": None,
            }
            output = self.mixture_model(collate_mixture_batch([item]), return_interactions=True, return_per_molecule_attribution=True)
            final = torch.sigmoid(output["logits"])[0]
            baseline = torch.sigmoid(output["baseline_logit"])[0]
            # Compute after sigmoid and after the model applies its learned interaction scale.
            difference = final - baseline
            matrix = output["interaction_matrix"][0].tolist()
            if not np.isfinite(matrix).all():
                raise RuntimeError("Invalid attention matrix")
            response["mixture"] = {
                "scores": scores(final), "baseline_scores": scores(baseline), "deltas": scores(difference),
                "attention_matrix": matrix,
                "attention_semantics": "last-layer head-averaged attention, not causal or signed interactions",
                "baseline_semantics": "sigmoid(logsumexp(component logits from the mixture checkpoint))",
                "raw_interaction_logits": output["interaction_logit"][0].tolist(),
                "context_scores": [scores(x) for x in torch.sigmoid(output["per_molecule_logits"])[0]],
            }
        response["inference_ms"] = round((time.perf_counter() - start) * 1000, 3)
        return response

    def metadata(self):
        return {"model_version": self.version, "dataset_kind": DATASET_KIND, "notice": NOTICE,
                "labels": [{"id": k, "en": k, "zh": v} for k, v in zip(FLAVOR_LABELS, LABELS_ZH)],
                "metrics": self.metrics, "feature_contract": self.manifest,
                "device": "cpu", "parameter_count": sum(p.numel() for p in self.mixture_model.parameters())}
