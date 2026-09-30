# MixtureFlavorNet Usage Guide

## Scope

This repository now contains a complete synthetic benchmark implementation of the mixture-flavor extension described in:

- `MIXTURE_FLAVOR_SPEC.md`
- `MultiMolecule_FlavorGNN_Technical_Analysis.md`
- the local `flavorgnn/` backbone project

The implementation reuses the existing `FlavorDMPNN` backbone, adds a mixture-level set transformer, trains a signed interaction head, and generates a reproducible dataset, tests, figures, and a report.

## Repository Areas

- `mixtureflavorgnn/`: mixture dataset, model, training, inference, and API modules
- `configs/mixture_base.yaml`: default config for workstation and laptop runs
- `configs/mixture_ubuntu_4090.yaml`: Ubuntu 22.04 and RTX 4090 oriented config
- `scripts/build_mixture_dataset.py`: dataset generation
- `scripts/train_mixture.py`: mixture model training
- `scripts/evaluate_mixture.py`: evaluation and figure generation
- `scripts/predict_mixture.py`: CLI inference
- `tests/test_mixture_*.py`: unit tests for the extension

## Dataset

The provided mixture dataset is a synthetic but structured benchmark built from the single-molecule demo dataset.

Files:

- `data/mixture_dataset.csv`
- `data/mixture_splits/train.csv`
- `data/mixture_splits/val.csv`
- `data/mixture_splits/test.csv`

Important columns:

- `components`: JSON list of component SMILES strings
- `component_labels`: JSON list of per-component labels
- `mixture_labels`: final mixture labels after additive and interaction rules
- `interaction_type`: `synergy`, `antagonism`, `mixed`, or `additive`
- `interaction_targets`: signed flavor deltas used for evaluation and auxiliary supervision
- `focus_component_index`, `focus_flavor`, `focus_expected_change`: counterfactual test metadata

## Recommended Environment

Current validated environment:

- Conda env: `flavor`
- Python: `3.13`
- PyTorch: CUDA-enabled

Future target deployment:

- Ubuntu 22.04
- Intel i9-14900K
- NVIDIA RTX 4090 24 GB
- 128 GB RAM

The code uses only portable Python, PyTorch, and pathlib APIs, so the same scripts work on Ubuntu once dependencies are installed.

## Commands

Build or rebuild the mixture dataset:

```bash
python scripts/build_mixture_dataset.py --config configs/mixture_base.yaml
```

Train the single-molecule backbone first:

```bash
python scripts/train.py --config configs/base.yaml
```

Train the mixture model:

```bash
python scripts/train_mixture.py --config configs/mixture_base.yaml --backbone-checkpoint artifacts/best_model.pt
```

Evaluate the mixture model and generate figures:

```bash
python scripts/evaluate_mixture.py --config configs/mixture_base.yaml --checkpoint artifacts/mixture/best_model.pt --split test
```

Run mixture inference from the CLI:

```bash
python scripts/predict_mixture.py --config configs/mixture_base.yaml --checkpoint artifacts/mixture/best_model.pt --smiles "NCC(=O)O" "NC(=O)CCS" "CC(=O)O"
```

Run tests:

```bash
pytest -q
```

## Ubuntu 22.04 / RTX 4090 Configuration

For the future production workstation, use:

```bash
python scripts/train.py --config configs/ubuntu_4090.yaml
python scripts/train_mixture.py --config configs/mixture_ubuntu_4090.yaml --backbone-checkpoint artifacts/best_model.pt
python scripts/evaluate_mixture.py --config configs/mixture_ubuntu_4090.yaml --checkpoint artifacts/mixture_4090/best_model.pt --split test
```

Notes:

- `configs/mixture_ubuntu_4090.yaml` raises `batch_size`, `num_workers`, warmup, and total mixture count.
- AMP is enabled by default.
- The mixture backbone learning rate is scaled down relative to the interaction head.

## Test Cases

The test suite covers:

- CSV parsing and padding of variable-length mixtures
- forward-pass correctness of `MixtureFlavorNet`
- metric calculation for multilabel and interaction-specific outputs

These are lightweight smoke and regression tests, not long-running end-to-end benchmarks.
