"""Rebuild deployment fixtures only after deliberately validating replacement weights.

Uses the copied original predictors as the reference, never the web adapter.
"""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / ".python-deps"), str(ROOT / "backend"), str(ROOT / "vendor/flavorgnn")]
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch
from rdkit import Chem, rdBase
from flavorgnn.constants import FLAVOR_LABELS
from flavorgnn.inference import FlavorPredictor
from mixtureflavorgnn.inference import MixturePredictor


def main():
    vendor = ROOT / "vendor/flavorgnn"
    os.chdir(vendor)
    torch.set_num_threads(2)
    single, mixture = FlavorPredictor(), MixturePredictor()
    inputs = ["CCO", "CC(=O)O"]
    one = single.predict(inputs, top_k=24, threshold=0)
    mix = mixture.predict(inputs, top_k=24, threshold=0, include_interaction_matrix=True)
    reference = {"source": "Copied original FlavorPredictor and MixturePredictor; CPU, RDKit features", "smiles": inputs,
                 "single_scores": [[next(f["probability"] for f in m["flavors"] if f["label"] == label) for label in FLAVOR_LABELS] for m in one],
                 "mixture_scores": [next(f["probability"] for f in mix["mixture_prediction"]["flavors"] if f["label"] == label) for label in FLAVOR_LABELS]}
    (vendor / "reference-predictions.json").write_text(json.dumps(reference, indent=2), encoding="utf-8")
    tracked = list(vendor.rglob("*.py")) + [vendor / p for p in ["configs/base.yaml", "configs/mixture_base.yaml", "artifacts/best_model.pt", "artifacts/mixture/best_model.pt", "reference-predictions.json"]]
    manifest = {"rdkit_version": rdBase.rdkitVersion, "reference_torch_version": torch.__version__, "feature_dimensions": [73, 14, 16],
                "labels": FLAVOR_LABELS, "dataset_kind": "synthetic_benchmark", "provenance": "Reference checks validate agreement with the copied local predictors; they do not establish sensory validity.",
                "sha256": {p.relative_to(vendor).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(tracked)}}
    (vendor / "model-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    examples = [
        {"id": "single", "name": {"zh": "从一个分子开始", "en": "Start with one molecule"}, "description": {"zh": "乙醇 · 单分子风味预测", "en": "Ethanol · single-molecule prediction"}, "mode": "single", "molecules": [{"name": "Ethanol", "smiles": "CCO"}]},
        {"id": "mixture", "name": {"zh": "探索混合后的变化", "en": "Explore a mixture"}, "description": {"zh": "乙醇 + 乙酸 · 两组分混合物", "en": "Ethanol + acetic acid · two components"}, "mode": "mixture", "molecules": [{"name": "Ethanol", "smiles": "CCO"}, {"name": "Acetic acid", "smiles": "CC(=O)O"}]},
        {"id": "batch", "name": {"zh": "比较多个分子", "en": "Compare molecules"}, "description": {"zh": "乙醇、乙酸、苯酚 · 分别预测", "en": "Ethanol, acetic acid, phenol · individual predictions"}, "mode": "batch", "molecules": [{"name": "Ethanol", "smiles": "CCO"}, {"name": "Acetic acid", "smiles": "CC(=O)O"}, {"name": "Phenol", "smiles": "c1ccccc1O"}]},
    ]
    for example in examples:
        smiles = [x["smiles"] for x in example["molecules"]]
        example["reference_result"] = mixture.predict(smiles, top_k=24, threshold=0) if example["mode"] == "mixture" else single.predict(smiles, top_k=24, threshold=0)
        example["dataset_kind"] = "synthetic_benchmark"
    (ROOT / "examples/examples.json").write_text(json.dumps({"examples": examples}, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "frontend/public/template.csv").write_text("name,smiles\nEthanol,CCO\nAcetic acid,CC(=O)O\n", encoding="utf-8-sig")
    writer = Chem.SDWriter(str(ROOT / "frontend/public/example.sdf"))
    for row in examples[1]["molecules"]:
        mol = Chem.MolFromSmiles(row["smiles"]); mol.SetProp("_Name", row["name"]); writer.write(mol)
    writer.close()
    print("Prepared manifest, reference predictions and three real-model examples in the independent project.")


if __name__ == "__main__":
    main()
