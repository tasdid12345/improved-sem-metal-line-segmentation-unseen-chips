"""Synthetic SEM-style IC segmentation: MLP on raw 7x7 patches (baseline) vs
per-image-normalised LightGBM with multi-scale filter features (improved).
Main metric: mean foreground IoU over metal lines and vias on unseen chips."""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import lightgbm as lgb

SEED = 42
SEEDS = [42, 43, 44]
H = W = 256
N_CHIPS, N_TEST_CHIPS = 10, 3          # 7 train chips, 3 unseen test chips (30% test)
N_TRAIN_PIX = 100_000                  # pixel subsample shared by both models
QUESTION = ("Can a per-image-normalised LightGBM pixel classifier with multi-scale filter "
            "features segment metal lines and vias on unseen synthetic SEM-style chips "
            "better than a standard MLP pixel classifier trained on raw patches?")
METHOD = ("Synthetic chips (parallel lines, square vias, blur, noise, per-chip gain/offset); "
          "chip-level train/test split; pixel classifiers scored by mean foreground IoU.")
NOTES = ("Synthetic data generated in code, not the paper's annotated SEM scans; numbers are not "
         "comparable to the paper's ESD error rates. Split is by chip (7 train, 3 unseen). "
         "Only LightGBM num_leaves is tuned, by 3-fold chip-grouped CV on training pixels; "
         "the MLP is untuned. Seeds 42-44 vary model initialisation/bagging only; the data "
         "and pixel subsample are fixed. The test set is used once per seed for both models "
         "with no selection on it. No downloads, so no retry logic is needed.")


def make_chip(rng):
    """Return (image float32, label uint8: 0=background, 1=metal line, 2=via)."""
    pitch = int(rng.integers(14, 28))
    width = int(rng.integers(5, pitch - 3))
    offset = int(rng.integers(0, pitch))
    lab = np.zeros((H, W), np.uint8)
    line_rows = ((np.arange(H) - offset) % pitch) < width   # parallel horizontal lines
    lab[line_rows, :] = 1
    cy = np.arange(offset + width // 2, H, pitch)            # line centre rows
    for _ in range(int(rng.integers(15, 30))):                # square vias on lines
        s = int(rng.integers(4, width))
        yc = int(rng.choice(cy))
        x0 = int(rng.integers(0, W - s))
        y0 = yc - s // 2
        lab[y0:y0 + s, x0:x0 + s] = 2
    img = np.where(lab == 2, 0.90, np.where(lab == 1, 0.65, 0.35))
    img = ndi.gaussian_filter(img, rng.uniform(1.0, 2.0))     # blur
    img = img + rng.normal(0, rng.uniform(0.04, 0.08), img.shape)  # noise
    gain, off = rng.uniform(0.6, 1.4), rng.uniform(-0.25, 0.25)    # per-chip brightness/contrast
    return (gain * img + off).astype(np.float32), lab


def raw_patch_feats(img, r=3):
    """Baseline: raw 7x7 intensity patch around every pixel (no normalisation)."""
    p = np.pad(img, r, mode="reflect")
    return sliding_window_view(p, (2 * r + 1, 2 * r + 1)).reshape(H * W, -1)


def enhanced_feats(img):
    """Improved: per-image normalised intensity plus a multi-scale filter bank."""
    z = (img - img.mean()) / (img.std() + 1e-6)              # per-image normalisation
    f = [z]
    f += [ndi.gaussian_filter(z, s) for s in (1, 2, 4, 8)]   # Gaussian smoothing
    f += [ndi.gaussian_gradient_magnitude(z, s) for s in (1, 2)]
    f += [ndi.gaussian_laplace(z, s) for s in (2, 4)]
    for w in (7, 15):                                        # local mean and std
        m = ndi.uniform_filter(z, w)
        f += [m, np.sqrt(np.maximum(ndi.uniform_filter(z * z, w) - m * m, 0))]
    f += [ndi.minimum_filter(z, 5), ndi.maximum_filter(z, 5)]
    return np.stack([a.ravel() for a in f], axis=1).astype(np.float32)


def evaluate(y, p):
    """Mean foreground IoU (lines, vias), pixel error rate and macro foreground F1."""
    ious = [np.sum((y == c) & (p == c)) / max(np.sum((y == c) | (p == c)), 1) for c in (1, 2)]
    return {"miou": float(np.mean(ious)), "error": float(np.mean(y != p)),
            "f1": float(f1_score(y, p, labels=[1, 2], average="macro"))}


def make_mlp(seed):
    return make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=60, early_stopping=True, random_state=seed))


def make_lgbm(num_leaves, seed):
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=num_leaves,
                              class_weight="balanced", subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, random_state=seed, verbose=-1)


def main():
    rng = np.random.default_rng(SEED)
    chips = [make_chip(rng) for _ in range(N_CHIPS)]
    n_tr = N_CHIPS - N_TEST_CHIPS
    tr_chips, te_chips = chips[:n_tr], chips[n_tr:]          # chip-level split
    print(f"generated {N_CHIPS} chips: {n_tr} train, {N_TEST_CHIPS} unseen test")

    # Subsample training pixels once (shared by both models)
    y_all = np.concatenate([l.ravel() for _, l in tr_chips])
    grp = np.repeat(np.arange(n_tr), H * W)
    idx = np.random.default_rng(SEED).choice(len(y_all), N_TRAIN_PIX, replace=False)
    y_tr, g_tr = y_all[idx], grp[idx]
    y_te = np.concatenate([l.ravel() for _, l in te_chips])
    if not ((y_te == 1).any() and (y_te == 2).any()):
        print("test set lacks foreground classes"); sys.exit(1)

    def build(fn, imgs):
        return np.concatenate([fn(im) for im in imgs])
    tr_imgs, te_imgs = [im for im, _ in tr_chips], [im for im, _ in te_chips]
    Xb_tr = build(raw_patch_feats, tr_imgs)[idx]
    Xi_tr = build(enhanced_feats, tr_imgs)[idx]
    Xb_te, Xi_te = build(raw_patch_feats, te_imgs), build(enhanced_feats, te_imgs)
    print(f"features: baseline {Xb_tr.shape[1]}, improved {Xi_tr.shape[1]}")

    # Tune LightGBM num_leaves with chip-grouped CV on training pixels only
    best = None
    for nl in (15, 63):
        cv = [evaluate(y_tr[b], make_lgbm(nl, SEED).fit(Xi_tr[a], y_tr[a]).predict(Xi_tr[b]))["miou"]
              for a, b in GroupKFold(3).split(Xi_tr, y_tr, g_tr)]
        print(f"CV num_leaves={nl}: mIoU={np.mean(cv):.4f}")
        if best is None or np.mean(cv) > best[1]:
            best = (nl, float(np.mean(cv)))
    NUM_LEAVES = best[0]
    print(f"selected num_leaves={NUM_LEAVES}")

    # Final fit and single test evaluation per seed, identical protocol for both models
    models = {"baseline": (make_mlp, Xb_tr, Xb_te),
              "improved": (lambda s: make_lgbm(NUM_LEAVES, s), Xi_tr, Xi_te)}
    runs = {k: [] for k in models}
    for seed in SEEDS:
        for name, (mk, Xtr, Xte) in models.items():
            t0 = time.time()
            m = mk(seed).fit(Xtr, y_tr)
            t = time.time() - t0
            r = evaluate(y_te, m.predict(Xte))
            r["time"] = t
            runs[name].append(r)
            print(f"seed {seed} {name}: mIoU={r['miou']:.4f} err={100*r['error']:.2f}% "
                  f"F1={r['f1']:.4f} train={t:.1f}s")

    def agg(name, key):
        v = np.array([r[key] for r in runs[name]])
        return float(v.mean()), float(v.std())

    b_m, b_s = agg("baseline", "miou")
    i_m, i_s = agg("improved", "miou")
    metrics = {
        "baseline_score": b_m, "improved_score": i_m, "improvement": i_m - b_m,
        "baseline_score_std": b_s, "improved_score_std": i_s,
        "baseline_f1": agg("baseline", "f1")[0], "improved_f1": agg("improved", "f1")[0],
        "baseline_error_rate_pct": 100 * agg("baseline", "error")[0],
        "improved_error_rate_pct": 100 * agg("improved", "error")[0],
        "baseline_train_time_s": agg("baseline", "time")[0],
        "improved_train_time_s": agg("improved", "time")[0],
        "improved_cv_miou": best[1], "improved_best_num_leaves": NUM_LEAVES,
    }
    if not all(np.isfinite(v) for v in metrics.values()):
        print("non-finite metrics"); sys.exit(1)

    results = {
        "question": QUESTION, "method": METHOD,
        "dataset": {"name": "Synthetic SEM-style IC chips (generated in code)",
                    "source_url": "https://arxiv.org/abs/2603.16548",
                    "n_rows": int(len(y_all) + len(y_te)), "n_train": int(len(y_tr)),
                    "n_test": int(len(y_te)), "same_as_paper": False,
                    "baseline_is_papers_method": False},
        "metric_name": "mean foreground IoU (metal lines, vias) on unseen chips",
        "higher_is_better": True,
        "baseline": "scikit-learn MLPClassifier on raw 7x7 intensity patches (standardised, untuned)",
        "improved": ("LightGBM pixel classifier on per-image-normalised intensity plus multi-scale "
                     "Gaussian, gradient, Laplacian and local-statistics features, balanced class "
                     "weights, num_leaves chosen by chip-grouped CV"),
        "metrics": metrics, "notes": NOTES,
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["baseline MLP", "improved LGBM"], [b_m, i_m], yerr=[b_s, i_s], capsize=5,
           color=["grey", "tab:blue"])
    ax.set_ylabel("mean foreground IoU (unseen chips)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig("figures/miou_bar.png", dpi=120)
    print("wrote results.json and figures/miou_bar.png")


if __name__ == "__main__":
    main()
