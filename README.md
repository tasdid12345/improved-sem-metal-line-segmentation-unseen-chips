# Synthetic SEM-style IC segmentation: MLP baseline vs. LightGBM with robust normalisation

## Question
Can a per-image-normalised LightGBM pixel classifier with multi-scale filter features segment metal lines and vias on unseen synthetic SEM-style chips better than a standard MLP pixel classifier trained on raw patches?

## Data
Synthetic 256x256 chips generated in code (parallel metal lines, square vias, blur, noise, per-chip gain/offset). 10 chips are generated: 7 for training and 3 unseen for testing. The split is by chip, so no test chip contributes pixels to training. Labels: 0 background, 1 metal line, 2 via.

## Methods
- **Baseline:** scikit-learn `MLPClassifier` (64, 32 hidden units, early stopping) on standardised raw 7x7 intensity patches. Untuned.
- **Improved:** LightGBM on features from a per-image percentile normalisation (2nd to 98th percentile), which is invariant to the per-chip gain and offset. The feature bank adds Gaussian smoothing, gradient magnitude, Laplacian of Gaussian, local mean and std, and min/max filters. Class weights are balanced. `num_leaves` (15 or 63) is chosen by 3-fold chip-grouped cross-validation on training pixels only.
- Training uses a fixed 100k-pixel subsample shared by both models. Seeds 42, 43 and 44 vary model initialisation only. The test set is used once per seed and is never used for tuning.

## Metric
Mean foreground IoU over metal lines and vias on the unseen chips. Higher is better. Pixel error rate and macro foreground F1 are also reported.

## Caveats
- The data is synthetic and generated in code. It is not the paper's annotated SEM data, so the numbers are not comparable to the paper's results.
- The earlier mean/std z-scoring of the improved model gave a large gap between cross-validated mIoU (about 0.87) and test mIoU (about 0.33). The percentile normalisation is intended to close that gap. This has not been confirmed by a run in this revision.

## Outputs
- `results.json`: metrics, baseline and improved scores, and notes.
- `figures/miou_bar.png`: bar chart of mean IoU with standard deviation across seeds.

## Reference

This project is motivated by the following paper:

> C Gehrmann, J Ricker, S Damm, D Cheng (2026). *SAMSEM--A Generic and Scalable Approach for IC Metal Line Segmentation*. arXiv preprint arXiv. https://arxiv.org/abs/2603.16548

Paper author: Prof. Dr. Asja Fischer, Ruhr-Universität Bochum (asja.fischer@rub.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
