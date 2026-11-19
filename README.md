# Robust Pixel-Wise RGB Classification under Poisson Noise: TEG–DCT–XGBoost

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/interface-Jupyter-orange" alt="Jupyter Notebook">
  <img src="https://img.shields.io/badge/model-XGBoost-red" alt="XGBoost">
  <img src="https://img.shields.io/badge/status-research%20prototype-yellow" alt="Research prototype">
</p>

<p align="center">
  <b>M. A. Rybnytskyi</b> ·
  <a href="https://dict.khai.edu/">
    Department of Information and Communication Technologies
  </a>,
  National Aerospace University «Kharkiv Aviation Institute»
</p>

## 📌 1. Overview

This repository contains an experimental Jupyter Notebook for studying the robustness of pixel-wise RGB image classification under signal-dependent Poisson noise.

The proposed lightweight framework combines:

- raw RGB pixel values;
- Texture–Entropy–Gradient (TEG) features;
- local Discrete Cosine Transform (DCT) ring-energy features;
- the Anscombe variance-stabilizing transform;
- an XGBoost multiclass classifier.

The primary objective is to determine whether compact local frequency-domain features improve classification robustness when the input RGB image is progressively degraded by Poisson noise.

The study also evaluates whether variance stabilization should be applied only to the DCT branch or to the complete RGB, TEG, and DCT feature pipeline.

---

## 🎯 2. Research Objectives

The experimental framework addresses the following research questions:

1. How strongly does Poisson noise affect the classification accuracy of a baseline RGB–TEG–XGBoost pipeline?
2. Does the addition of local DCT ring-energy features improve classification quality?
3. Does the Anscombe variance-stabilizing transform increase the robustness of frequency-domain features?
4. Is it beneficial to apply variance stabilization to all feature branches?
5. How does XGBoost redistribute feature importance between RGB, TEG, and DCT features as the noise level increases?
6. How does the amount of labeled training data affect each pipeline?
7. What computational overhead is introduced by DCT and variance-stabilizing preprocessing?

---

## ⚙️ 3. Processing Framework

The complete processing workflow is:

```text
RGB image
   │
   ├── Poisson noise simulation
   │
   ├── Optional variance stabilization
   │      ├── Forward Anscombe transform
   │      ├── Gaussian smoothing
   │      └── Inverse Anscombe transform
   │
   ├── Raw RGB values
   ├── TEG feature extraction
   ├── Local DCT ring-energy extraction
   │
   ├── Feature concatenation
   │
   └── XGBoost multiclass classification
          │
          ├── F1-Macro evaluation
          ├── feature-importance analysis
          └── full-scene classification map