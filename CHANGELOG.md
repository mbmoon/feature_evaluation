---

## `CHANGELOG.md`

```markdown
# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Planned

- Add a noise-aware U-Net semantic-segmentation baseline.
- Add a neural denoising and segmentation baseline.
- Add cross-noise evaluation with a fixed model tested on unseen noise levels.
- Add repeated experiments with multiple random seeds.
- Add cross-scene evaluation.

## [0.1.0] - 2026-09-21

### Added

- Initial research implementation of a pixel-wise RGB image classification framework based on TEG, local DCT ring-energy features, and XGBoost.
- Poisson-noise evaluation for `Peak` values from 100 to 5.
- Anscombe-based variance-stabilizing preprocessing.
- Four ablation-study pipelines:
  - `Raw TEG`;
  - `TEG + DCT`;
  - `VSP(DCT) + TEG`;
  - `VSP(TEG + DCT)`.
- Training-data-efficiency experiments from 1% to 100% of the available labeled pixels.
- F1-Macro evaluation and computational profiling.
- XGBoost feature-importance analysis using normalized `total_gain`.
- DCT window-size optimization.
- Full-scene classification map generation and Ground Truth comparison.
- Export of color classification maps and numerical class-label arrays.

### Current Status

- The classical TEG–DCT–XGBoost experimental pipeline is implemented and operational.
- The matched-noise robustness, feature-importance, timing, and qualitative map experiments have been completed.
- Neural-network baselines and cross-noise generalization experiments remain planned for future development.