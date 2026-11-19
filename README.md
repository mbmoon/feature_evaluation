# Robust Pixel-Wise Remote Sensing Classification: Multi-Criteria Feature Selection, Dyadic Quadrant Polling (DQP), and Machine Learning Benchmarks

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/interface-Jupyter-orange" alt="Jupyter Notebook">
  <img src="https://img.shields.io/badge/models-XGBoost%20%7C%20MLP%20%7C%201D--CNN%20%7C%20TabNet-purple" alt="Models">
  <img src="https://img.shields.io/badge/data-Sentinel--2-darkgreen" alt="Sentinel-2">
  <img src="https://img.shields.io/badge/status-active%20research-brightgreen" alt="Research Status">
</p>

<p align="center">
  <b>Maksym A. Rybnytskyi</b> ·
  <a href="https://orcid.org/0009-0000-1299-1604">ORCID: 0009-0000-1299-1604</a><br>
  <a href="https://dict.khai.edu/">
    Department of Information and Communication Technologies
  </a>,
  National Aerospace University «Kharkiv Aviation Institute»
</p>

---

## 📌 1. Overview & Research Scope

This repository provides an open-source, publication-grade research and benchmarking framework for **robust pixel-wise classification of remote sensing imagery (Sentinel-2)** under severe sensor and environmental noise.

High-resolution pixel-wise segmentation in earth observation often suffers from:
1. **Severe noise degradation** (additive Gaussian sensor thermal noise, signal-dependent Poisson photon-counting noise, and atmospheric/transmission impulse noise).
2. **Homogeneous region instability**, where subtle noise fluctuations cause salt-and-pepper misclassifications in flat agricultural fields or water bodies.
3. **High computational cost** of spatial, texture, and frequency feature extraction when applied to large satellite tiles.

### Key Contributions of this Framework:
* **Novel Spatial Structural Descriptor — Dyadic Quadrant Polling (DQP):** An asymmetric directional feature extractor designed to suppress noise-induced classification artifacts in homogeneous zones and preserve critical land-cover boundaries.
* **Multi-Criteria Feature Selection (Mode B / Utility Function):** A Pareto-efficient feature selection strategy that maximizes classification information gain while penalizing feature correlation/redundancy and per-scene feature extraction time ($P_{\text{time}}(t_{f_i}) = 1 + \alpha \cdot t_{f_i}$).
* **Cross-Architecture Machine Learning Benchmark:** Rigorous comparison across four diverse classifier paradigms:
  1. **XGBoost** (standard: 200 trees, depth 8; lightweight: 50 trees, depth 5);
  2. **Base NN / MLP** (DeepHyperX tabular neural network baseline);
  3. **1D-CNN** (Spectral/feature convolutional network based on Hu et al.);
  4. **TabNet** (Attentive transformer with sequential multi-step attention by Arik & Pfister, Google Cloud AI).
* **Cross-Noise Evaluation (11 Scenarios):** Systematic robustness evaluation across Additive White Gaussian Noise (AWGN), Poisson photon noise, and Salt & Pepper impulse noise.
* **Production-Grade Hardware Benchmarking Protocol:** Model training accelerated on GPU (Apple Silicon MPS / CUDA) combined with pure CPU full-scene ($512 \times 512 = 262,144$ pixels) inference profiling for real-world deployment.

---

## 🏗️ 2. Methodological Architecture

The complete image processing, feature engineering, and classification pipeline:

```text
Raw Sentinel-2 Scene (512x512 RGB)
   │
   ├── Noise Degradation Simulation (11 Scenarios)
   │      ├── AWGN (σ² ∈ {25, 50, 100})
   │      ├── Poisson (Peak ∈ {100, 40, 20, 10, 5})
   │      └── Impulse Salt & Pepper (1%, 5%, 20%)
   │
   ├── Spatial Isolation Protocol (Spatial Train / Val / Test Masking)
   │
   ├── Full 33-D Feature Pool Extraction:
   │      ├── Raw Spectral Channels (3): R, G, B
   │      ├── Structural TEG Descriptors (15): Texture, Entropy, Sobel Gradient
   │      ├── 2D-DCT Ring-Energy Frequency Bands (9): Low, Mid, High Rings
   │      └── Dyadic Quadrant Polling (DQP) (6): Asymmetric Quadrant Metrics
   │
   ├── Multi-Criteria Feature Selection (Mode B):
   │      ├── 1% Spatial Calibration Probing (Fast Gain Estimation)
   │      ├── Inter-Feature Redundancy & Correlation Discounting
   │      └── Feature Extraction Time Penalization: P_time(t_fi) = 1 + α * t_fi
   │          └── Yields Adaptive Compact Space: K* ∈ [9 .. 14] optimal features
   │
   └── Comprehensive Multi-Classifier Benchmark:
          ├── XGBoost (200 trees, max_depth=8)
          ├── Fast XGBoost (50 trees, max_depth=5)
          ├── Base NN / MLP (Linear-ReLU-Dropout-Linear)
          ├── 1D-CNN (Conv1D-BatchNorm-ReLU-MaxPool-FC)
          └── TabNet (Sequential Multi-Step Attention Transformer)
                 │
                 ├── Macro-averaged F1-Score (F1-Macro)
                 ├── Computational Latency: T_ext, T_fit, T_inf
                 └── Full-Scene Segmentation Maps & Error Analysis
```

---

## 🔬 3. Feature Engineering: The 33-D Candidate Pool

The candidate feature space integrates multi-domain descriptors:

| Feature Subset | Dimensionality | Domain | Description |
| :--- | :---: | :--- | :--- |
| **RAW Spectral** | 3 | Spectral | Direct R, G, B normalized reflectance intensities. |
| **TEG (Texture-Entropy-Gradient)** | 15 | Spatial & Statistical | Multi-channel local entropy in sliding windows, Sobel spatial gradients ($G_x, G_y$), and Local Binary Patterns (LBP). |
| **Radial 2D-DCT** | 9 | Frequency | Local Discrete Cosine Transform ring-energy features capturing low-, mid-, and high-frequency spectral signatures. |
| **DQP (Dyadic Quadrant Polling)** | 6 | Directional Structural | Directional quadrant sampling measuring gradient and intensity asymmetry across dyadic quadrants to stabilize flat regions. |
| **Total Pool** | **33** | Combined | Comprehensive spatio-frequency-structural representation. |

### Dyadic Quadrant Polling (DQP)
The DQP descriptor partitions the local neighborhood into four dyadic quadrants ($Q_1, Q_2, Q_3, Q_4$). By comparing directional energy dispersion and quadrant-level mean deviations, DQP provides spatial context that distinguishes true land-cover boundaries from isolated noise spikes.

<p align="center">
  <img src="dqp_geometry.png" alt="DQP Geometry" width="450">
  <br>
  <i>Figure 1: Geometric partitioning of the local pixel neighborhood into dyadic quadrants (DQP).</i>
</p>

---

## ⚖️ 4. Multi-Criteria Feature Selection (Mode B)

Traditional feature selection solely based on Information Gain ($G_{f_i}$) tends to select redundant, correlated features and computationally expensive descriptors. Mode B formulates feature selection as an optimization problem:

$$U(f_i) = \frac{G_{f_i} \cdot \prod_{j \in S} (1 - |\rho_{i,j}|)}{P_{\text{time}}(t_{f_i})}$$

Where:
* $G_{f_i}$ is the Information Gain estimated via fast probing on a 1% spatial calibration sample;
* $\rho_{i,j}$ is the Pearson correlation coefficient between candidate feature $f_i$ and already selected feature $f_j \in S$;
* $P_{\text{time}}(t_{f_i}) = 1 + \alpha \cdot t_{f_i}$ penalizes slow descriptors (e.g., sliding-window local entropy) according to scene extraction time $t_{f_i}$ and penalty parameter $\alpha \in [10; 25]$.

---

## 📊 5. Summary Experimental Results

Below is a representative benchmark comparison of classification accuracy (F1-score Macro) across noise scenarios on Sentinel-2 scene **SS1**:

| Noise Scenario | Feature Set | Dim | XGBoost (200, 8) | Fast XGB (50, 5) | Base NN / MLP | 1D-CNN | TabNet |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AWGN ($\sigma^2=25$)** | RAW | 3 | 0.8356 | 0.8452 | 0.8525 | **0.8681** | 0.8411 |
| | Full Pool (with DQP) | 33 | 0.9410 | 0.9363 | 0.9519 | **0.9527** | 0.9520 |
| | Mode B (Selected) | 11 | 0.9403 | 0.9388 | 0.9433 | **0.9462** | 0.9394 |
| **AWGN ($\sigma^2=100$)** | RAW | 3 | 0.7772 | 0.7965 | **0.8112** | 0.7910 | 0.7861 |
| | Full Pool (with DQP) | 33 | 0.9374 | 0.9321 | **0.9497** | 0.9431 | 0.9456 |
| | Mode B (Selected) | 9 | 0.9415 | 0.9367 | 0.9397 | 0.9484 | **0.9485** |
| **Poisson (Peak=40)** | RAW | 3 | 0.7133 | 0.7502 | 0.7521 | **0.7568** | 0.7515 |
| | Full Pool (with DQP) | 33 | 0.9312 | 0.9305 | **0.9487** | 0.9478 | 0.9421 |
| | Mode B (Selected) | 11 | 0.9360 | 0.9308 | 0.9335 | 0.9037 | **0.9450** |
| **Poisson (Peak=5)** | RAW | 3 | 0.5482 | **0.5547** | 0.5351 | 0.5309 | 0.5526 |
| | Full Pool (with DQP) | 33 | 0.8883 | 0.8932 | 0.9085 | 0.8972 | **0.9122** |
| | Mode B (Selected) | 14 | 0.8873 | 0.8933 | **0.9190** | 0.8979 | 0.9025 |
| **Impulse (S&P 5%)** | RAW | 3 | 0.8460 | **0.8516** | 0.8501 | 0.8238 | 0.8214 |
| | Full Pool (with DQP) | 33 | 0.9456 | 0.9475 | **0.9532** | 0.9447 | 0.9470 |
| | Mode B (Selected) | 14 | 0.9499 | 0.9474 | **0.9562** | 0.9467 | 0.9441 |

### Key Observations:
1. **DQP & Multi-Domain Gains:** Moving from RAW (3) to Full (33) features increases F1-Macro by **+10% to +36%**, with the largest gains observed under severe noise (Poisson Peak 5: F1 rises from 0.55 to 0.91).
2. **Mode B Efficiency:** Selecting only **9–14 features** with the proposed utility function retains **99.5%–100.5%** of the full 33-feature F1 accuracy while reducing feature extraction time from ~0.59 s down to **0.03–0.05 s** per scene (>10× speedup).
3. **Inference vs Training Trade-Off:** While neural networks (MLP, 1D-CNN) achieve high F1 accuracy, Fast XGBoost (50 trees, depth 5) achieves competitive accuracy with an inference latency of only **~45–50 ms** on CPU for an entire 262,144-pixel scene.

---

## 📂 6. Repository Structure

```text
├── README.md                           # Main project documentation and benchmark summary
├── CHANGELOG.md                        # Version history and detailed changelog
├── CITATION.cff                        # Citation File Format metadata
├── LICENSE                             # MIT License
├── requirements.txt                    # Python library dependencies
├── dqp_geometry.png                    # DQP quadrant partition diagram (PNG)
├── dqp_geometry.svg                    # DQP quadrant partition diagram (Vector SVG)
│
├── teg_dct_gain_final.ipynb            # ★ PRIMARY RESEARCH NOTEBOOK: Unified end-to-end pipeline,
│                                       #   DQP validation, Mode B selection, benchmarks, and tables
├── teg_dct_gain_superprediction.ipynb  # Multi-model training and checkpointing suite (MLP, 1D-CNN, TabNet)
├── teg_dct_gain_prediction.ipynb       # Exploratory prediction and feature analysis notebook
├── visualize DQP.ipynb                 # Visual and mathematical verification of DQP features
├── mutate_images_with_noise.ipynb      # Noise simulation suite (AWGN, Poisson, Impulse S&P)
├── prepare_train_and_test_masks.ipynb  # Spatial isolation protocol (Train/Val/Test spatial splits)
├── get_sentinel_2_RGB_images.ipynb     # Sentinel-2 STAC API data acquisition script
└── visualize_dataset.ipynb             # Ground truth and class distribution visualizer
```

---

## 🚀 7. Installation & Quick Start

### Prerequisites
* Python 3.10, 3.11, or 3.12
* Recommended hardware: Apple Silicon Mac (M1/M2/M3/M4 with MPS) or Linux with CUDA GPU

### Setup Environment
```bash
# 1. Clone repository
git clone https://github.com/mbmoon/feature_evaluation.git
cd feature_evaluation

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Running the Experiments
Launch Jupyter Lab or Notebook:
```bash
jupyter lab
```
Open **`teg_dct_gain_final.ipynb`**:
* **Cell 2:** Set `SCENE_NAME = "SS1"` or `SCENE_NAME = "SS2"`.
* Run all cells to reproduce the feature extraction, Mode B adaptive selection, model benchmarks, and final comparative tables.

---

## 📜 8. Citation

If you use this software, code, or experimental benchmarks in your research, please cite:

```bibtex
@software{rybnytskyi2026feature_evaluation,
  author       = {Rybnytskyi, Maksym A.},
  title        = {{Robust Pixel-Wise Remote Sensing Classification: Multi-Criteria Feature Selection, Dyadic Quadrant Polling (DQP), and Machine Learning Benchmarks}},
  month        = nov,
  year         = 2026,
  publisher    = {GitHub},
  version      = {1.0.0},
  url          = {https://github.com/mbmoon/feature_evaluation}
}
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
