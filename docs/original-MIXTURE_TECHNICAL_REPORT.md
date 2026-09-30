# MixtureFlavorNet Technical Report

## Abstract

This report documents the end-to-end implementation and evaluation of **MixtureFlavorNet**, a hierarchical multi-molecule flavor prediction system built from the local `flavorgnn` backbone and the two specification documents in the workspace. The final system combines a reused **FlavorDMPNN** molecular encoder with a **set transformer** interaction module and a signed **synergy-antagonism head**. On the held-out synthetic mixture benchmark, the model achieved **macro AUROC = 0.9736**, **macro average precision = 0.9326**, **macro F1 = 0.8798**, **synergy detection rate = 0.9289**, **antagonism detection rate = 0.9302**, and **counterfactual accuracy = 0.7789**. Relative to the single-molecule baseline checkpoint, the mixture system improved macro AUROC by **+0.2167**, macro average precision by **+0.3661**, and macro F1 by **+0.5120**.

## 1. Objective

The target problem is mixture-level flavor prediction over unordered sets of molecules. The implementation requirements were:

- preserve the original 24-label taxonomy from `flavorgnn`
- model cross-molecular interaction explicitly
- keep the code runnable in the current Conda environment and portable to Ubuntu 22.04 with an RTX 4090
- provide datasets, tests, usage documentation, and a reproducible technical report

## 2. Implementation Summary

The delivered code is located in the new `mixtureflavorgnn/` package and follows the architecture proposed in the specifications:

- **Stage 1: Molecular encoding**
  - shared `FlavorDMPNN` backbone from the local `flavorgnn` project
  - checkpoint initialization from `artifacts/best_model.pt`
  - projection from fused graph/global features to a 256-dimensional mixture embedding
- **Stage 2: Interaction encoding**
  - 4-layer set transformer with 8 attention heads
  - permutation-invariant reasoning over molecule embeddings
  - learned inducing-point pooling for mixture representation
- **Stage 3: Synergy-antagonism decoding**
  - additive baseline from per-molecule logits via masked `logsumexp`
  - signed interaction delta from an MLP over the pooled mixture embedding
  - final logits formed as `baseline + lambda * interaction_delta`

The model contains **5,867,106 trainable parameters**.

## 3. Data

### 3.1 Single-Molecule Backbone Data

The original local demo dataset contains **142** labeled single molecules. It was used to train the checkpoint that initializes the mixture encoder.

### 3.2 Mixture Benchmark

The mixture dataset was generated programmatically from the single-molecule corpus and saved to:

- `data/mixture_dataset.csv`
- `data/mixture_splits/train.csv`
- `data/mixture_splits/val.csv`
- `data/mixture_splits/test.csv`

Final dataset statistics:

| Quantity | Value |
|---|---:|
| Total mixtures | 4096 |
| Train mixtures | 3276 |
| Validation mixtures | 409 |
| Test mixtures | 411 |
| Interaction-bearing mixtures | 64.99% |
| Mean mixture size | 2.35 molecules |
| Median mixture size | 2 molecules |

The benchmark includes additive examples plus curated synthetic interaction rules for synergy, antagonism, and mixed behavior. Counterfactual metadata is stored per mixture to support removal-based evaluation.

## 4. Experimental Protocol

### 4.1 Current Runtime

The full pipeline was executed in the current `flavor` Conda environment. The observed runtime stack was:

- Python `3.13.7`
- PyTorch `2.11.0+cu126`
- CUDA available: `True`
- RDKit available: `True`

The repository test suite completed successfully with **7/7 tests passing**.

### 4.2 Target Deployment

For the future Ubuntu 22.04 workstation, the repository now includes:

- `configs/ubuntu_4090.yaml`
- `configs/mixture_ubuntu_4090.yaml`

These configurations raise worker count, batch size, and warmup length for a 24 GB RTX 4090.

### 4.3 Training Procedure

1. Train the single-molecule backbone with `scripts/train.py`.
2. Build the mixture dataset with `scripts/build_mixture_dataset.py`.
3. Train the mixture model with `scripts/train_mixture.py`.
4. Evaluate and generate figures with `scripts/evaluate_mixture.py`.

## 5. Results

### 5.1 Backbone vs. Mixture Model

| Metric | Single-Molecule Backbone | MixtureFlavorNet | Improvement |
|---|---:|---:|---:|
| Macro AUROC | 0.7569 | 0.9736 | +0.2167 |
| Macro Average Precision | 0.5665 | 0.9326 | +0.3661 |
| Macro F1 | 0.3679 | 0.8798 | +0.5120 |

The mixture model substantially outperformed the backbone-only reference, which is expected because the benchmark includes explicit interaction rules that a single-molecule predictor cannot represent.

### 5.2 Final Held-Out Mixture Metrics

| Metric | Value |
|---|---:|
| Macro AUROC | 0.9736 |
| Macro Average Precision | 0.9326 |
| Macro F1 | 0.8798 |
| Exact Match Ratio | 0.2798 |
| Hamming Loss | 0.0591 |
| Expected Calibration Error | 0.0680 |
| Synergy Detection Rate | 0.9289 |
| Antagonism Detection Rate | 0.9302 |
| Mean Lift Score | 1.2400 |
| Counterfactual Accuracy | 0.7789 |

### 5.3 Interpretation

- The **high AUROC and AP** indicate that the model learned the label structure of the synthetic interaction benchmark well.
- **Macro F1 = 0.8798** shows that thresholded multilabel predictions are strong, not just rank ordering.
- **Synergy and antagonism detection rates above 0.92** confirm that the signed interaction branch is recovering the intended interaction direction in most held-out cases.
- **Counterfactual accuracy = 0.7789** shows that removing a key molecule changes the predicted target flavor in the expected direction for most annotated counterfactual cases.
- **ECE = 0.0680** indicates useful but not perfect calibration; temperature scaling remains a sensible post-processing option.

## 6. Figures

### Figure 1. Training dynamics

![Training curves](artifacts/mixture_report/training_curves.png)

The optimization trace shows stable convergence with rising validation AUROC across epochs and no sign of catastrophic divergence after backbone unfreezing.

### Figure 2. Per-label discrimination

![Per-label metrics](artifacts/mixture_report/label_metrics.png)

Per-label AUROC and average precision remain consistently high across the 24-label taxonomy, indicating that improvements are not concentrated in only one or two labels.

### Figure 3. Calibration

![Calibration](artifacts/mixture_report/calibration.png)

The reliability diagram shows moderate overconfidence in some probability bins but overall acceptable calibration for a synthetic benchmark. This is consistent with the reported ECE of 0.0680.

### Figure 4. Interaction delta magnitude

![Interaction histogram](artifacts/mixture_report/interaction_delta_hist.png)

The delta histogram shows that the model is using the interaction branch actively. A useful next step is to tighten sparsity so additive examples induce smaller deltas.

## 7. Error Analysis and Limitations

### 7.1 Strengths

- Strong mixture-level discrimination and thresholded multilabel performance
- Successful signed interaction recovery
- Meaningful counterfactual behavior
- Full reproducibility through scripts, configs, and saved artifacts

### 7.2 Remaining Weaknesses

- The benchmark is **synthetic**, not sensory-panel ground truth. The model therefore demonstrates engineering validity and controlled interaction learning, not wet-lab validation.
- The non-interacting delta magnitude remains larger than ideal, indicating that the sparsity prior can be strengthened further.
- Exact match ratio is lower than ranking metrics because the label space is broad and many mixtures contain multiple simultaneously active flavors.

## 8. Reproducibility Assets

The following deliverables were created in the repository:

- `mixtureflavorgnn/` implementation package
- `configs/mixture_base.yaml`
- `configs/mixture_ubuntu_4090.yaml`
- `scripts/build_mixture_dataset.py`
- `scripts/train_mixture.py`
- `scripts/evaluate_mixture.py`
- `scripts/predict_mixture.py`
- `data/mixture_dataset.csv`
- `data/mixture_splits/`
- `artifacts/mixture/`
- `artifacts/mixture_report/`
- `MIXTURE_USAGE.md`

## 9. Conclusion

The requested project has been implemented and executed successfully. The resulting MixtureFlavorNet system is functional, reproducible, and materially stronger than the single-molecule baseline for the defined mixture benchmark. The code is portable to the specified Ubuntu 22.04 / RTX 4090 environment through dedicated config files, and the repository now includes the dataset, tests, evaluation figures, and operational documentation needed for continued experimentation.
