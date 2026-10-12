# Synthetic SEM-style IC segmentation: LightGBM filter features vs MLP on raw patches

Motivation: "SAMSEM--A Generic and Scalable Approach for IC Metal Line Segmentation" (2026) segments metal lines in SEM images of integrated circuits with a fine-tuned SAM2-based model and reports error rates both on chips it was fine-tuned on and on unseen chips. Its generalisation gap motivates asking whether a much simpler per-pixel classifier, given per-image normalised multi-scale features, can generalise better to unseen chips than a standard patch-based pixel classifier.

Question: Can a per-image-normalised LightGBM pixel classifier with multi-scale filter features segment metal lines and vias on unseen synthetic SEM-style chips better than a standard MLP pixel classifier trained on raw patches?

Dataset: synthetic grayscale SEM-style chips generated in code (parallel metal lines, square vias, blur, noise, per-chip brightness/contrast offset). The data is not the paper's own; its SEM scans are not publicly downloadable. Train and test chips are separate (chip-level split).

Baseline: scikit-learn MLPClassifier on raw 7x7 intensity patches, standing in for the paper's segmenter. It is not the paper's method.

Improved: LightGBM pixel classifier on per-image normalised intensity plus multi-scale Gaussian, gradient, Laplacian and local-statistics features, with balanced class weights and a small chip-grouped CV tuning of num_leaves.

How to run:

    pip install -r requirements.txt
    python analysis.py

Results: results are written to results.json (and figures/miou_bar.png).

## Reference

This project is motivated by the following paper:

> C Gehrmann, J Ricker, S Damm, D Cheng (2026). *SAMSEM--A Generic and Scalable Approach for IC Metal Line Segmentation*. arXiv preprint arXiv. https://arxiv.org/abs/2603.16548

Paper author: Prof. Dr. Asja Fischer, Ruhr-Universität Bochum (asja.fischer@rub.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
