# FlavorGNN (Demo Implementation)

Python project for **small-molecule flavor prediction with Graph Neural Networks**, implemented from the provided technical specification (`FlavorGNN_Technical_Spec.docx`).

This repository includes:
- End-to-end data pipeline (SMILES -> graph -> train/val/test split)
- Directed MPNN model with attention readout
- Training, evaluation, inference CLI
- FastAPI inference service
- Demo dataset + tests

## 1. Environment (Ubuntu 22.04 + RTX 4090)

Recommended:
- Python `3.11` or `3.12`
- Existing PyTorch CUDA environment (you said it is already installed)

Install extra dependencies:

```bash
pip install -r requirements.txt
```

If PyTorch is not installed yet:

```bash
pip install -r requirements-full.txt
```

## 2. Build Dataset and Splits

Generate demo dataset and split files:

```bash
python scripts/build_demo_dataset.py --make-splits
```

Outputs:
- `data/demo_flavor_dataset.csv`
- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`

## 3. Train

Base config:

```bash
python scripts/train.py --config configs/base.yaml
```

4090-oriented config:

```bash
python scripts/train.py --config configs/ubuntu_4090.yaml
```

Artifacts are written to `artifacts/` (best checkpoint, summaries, metrics).

## 4. Evaluate

```bash
python scripts/evaluate.py --config configs/base.yaml --checkpoint artifacts/best_model.pt --split test
```

## 5. Predict (CLI)

```bash
python scripts/predict.py --config configs/base.yaml --checkpoint artifacts/best_model.pt --smiles CCO CC(=O)O c1ccccc1O
```

PowerShell example (quote SMILES containing parentheses):

```powershell
python scripts/predict.py --config configs/base.yaml --checkpoint artifacts/best_model.pt --smiles "CCO" "CC(=O)O" "c1ccccc1O"
```

Or use JSON input:

```bash
python scripts/predict.py --config configs/base.yaml --checkpoint artifacts/best_model.pt --input-json examples/predict_samples.json
```

## 6. Run API

```bash
python scripts/run_api.py --host 0.0.0.0 --port 8000
```

API:
- `GET /v1/health`
- `GET /v1/flavor-labels`
- `POST /v1/predict`

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"smiles":["CCO","CC(=O)O"],"top_k":5,"threshold":0.5}'
```

## 7. Tests

```bash
pytest -q
```

If `pytest` resolves to another Python environment, run:

```bash
python -m pytest -q
```

## 8. Notes

- RDKit is optional. If available, featurization uses RDKit automatically.
- Without RDKit, a pure-Python fallback parser is used for demo usability.
- Demo dataset is synthetic and intended for functional validation.

Detailed design doc: see [PROJECT_DESIGN.md](PROJECT_DESIGN.md).
