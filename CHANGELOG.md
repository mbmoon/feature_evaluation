# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-11-19

### Added
- **Dyadic Quadrant Polling (DQP) Descriptor (6 features):** Novel asymmetric directional spatial feature extractor designed to suppress noise-induced misclassifications in homogeneous flat regions and stabilize edge transitions.
- **Multi-Criteria Feature Selection (Mode B):** Adaptive utility function $U(f_i)$ balancing Information Gain with correlation redundancy discounting and per-scene feature extraction time penalization ($P_{\text{time}} = 1 + \alpha \cdot t_{f_i}$).
- **Comprehensive Multi-Classifier Benchmark Suite:**
  - Standard **XGBoost** (200 trees, max_depth=8);
  - Lightweight **Fast XGBoost** (50 trees, max_depth=5) for low-latency edge deployment;
  - Tabular Deep Learning Baseline **Base NN / MLP** (DeepHyperX architecture);
  - Convolutional Baseline **1D-CNN** (Hu et al. spectral/feature 1D convolution);
  - Sequential Multi-Step Attention Transformer **TabNet** (Arik & Pfister, Google Cloud AI).
- **Comprehensive Multi-Noise Stress Testing (11 Scenarios across 3 Families):**
  - Additive White Gaussian Noise (AWGN, $\sigma^2 \in \{25, 50, 100\}$);
  - Signal-dependent Poisson Noise ($\text{Peak} \in \{100, 40, 20, 10, 5\}$);
  - Impulse Salt & Pepper Noise ($\text{S\&P} \in \{1\%, 5\%, 20\%\}$).
- **Production Hardware Profiling Protocol:** Hybrid evaluation combining GPU training acceleration (Apple Silicon MPS / CUDA) with pure CPU full-scene ($512 \times 512 = 262,144$ pixels) inference latency profiling.
- **Cross-Scene Evaluation:** Comprehensive validation across Sentinel-2 scenes `SS1` and `SS2` with spatial isolation masking to prevent data leakage.
- **Automated Comparative Reporting:** Generation of unified comparative Markdown and LaTeX tables for classification accuracy (F1-score Macro) and computational timings ($T_{\text{ext}}$, $T_{\text{fit}}$, $T_{\text{inf}}$).
- **High-Resolution Segmentation Map Visualizer (Cell 10d):** Individual mask generation for qualitative Ground Truth vs Base 27 vs Full 33 (+DQP) comparisons under severe noise conditions.

### Changed
- Unified the end-to-end experimental research pipeline into `teg_dct_gain_final.ipynb`.
- Expanded the candidate feature pool from 27 base features (RGB + TEG + DCT) to a 33-dimensional space (+6 DQP features).
- Cleaned and restructured the repository, updating `.gitignore` to prevent tracking virtual environments, OS metadata, and intermediate checkpoints.

---

## [0.1.0] - 2026-09-21

### Added
- Initial experimental prototype combining raw RGB, TEG features, and local 2D-DCT ring energy with XGBoost.
- Preliminary Poisson noise simulation (Peak 100 to 5) and Anscombe variance-stabilizing transform evaluation.
- Basic F1-Macro metric tracking and feature importance profiling.
