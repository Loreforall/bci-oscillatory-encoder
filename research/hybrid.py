"""Calibrated multiple-kernel regression on intact visual and oscillatory blocks."""
import numpy as np

from .cove import fit_coupling
from .experiment import design, penalties, ridge, scores, split

# Positive mixtures keep the similarity operator positive semidefinite.
MIXTURES = {
    "visual_kernel": (1., 0., 0.),
    "detail_kernel": (.55, .45, 0.),
    "hybrid_light": (.90, .04, .06),
    "hybrid_state": (.94, 0., .06),
    "hybrid": (.55, .30, .15),
    "hybrid_more_state": (.45, .25, .30),
}
ERP_BINS = 30


def components(visuals, slopes, fps=24):
    """Keep the visual bypass separate from bounded, complex state."""
    return (
        np.concatenate([v.conventional for v in visuals]),
        np.concatenate([np.column_stack([v.detail, 4*v.spectrum, v.pixels])
                        for v in visuals]),
        np.concatenate(design(visuals, slopes, fps, "advanced")),
    )


def block_kernel(x, train, width):
    mean = x[train].mean(axis=0)
    scale = x[train].std(axis=0).clip(.01)
    centered = (x - mean) / scale
    anchors = centered[train]
    square = (centered**2).sum(axis=1)
    distances = np.maximum(0., square[:, None] + square[train] - 2 * centered @ anchors.T)
    distances /= x.shape[1]
    return np.exp(-distances / width), dict(mean=mean, scale=scale, anchors=anchors)


def onset_smoother(strength):
    """Tikhonov smoothing on post-onset ERP bins with natural edges."""
    second = np.diff(np.eye(ERP_BINS), n=2, axis=0)
    return np.linalg.solve(np.eye(ERP_BINS) + strength * second.T @ second, np.eye(ERP_BINS))


def fit(visuals, target, groups, seed, *, erp=True, require_state=False):
    train, val, test = split(groups, seed)
    slopes = fit_coupling([visuals[index] for index in train])
    blocks = components(visuals, slopes)
    y = target.reshape(len(groups), -1)
    reference = y[train].mean(axis=0)
    rng = np.random.default_rng(seed)
    x = blocks[0]
    hidden = np.tanh(((x - x[train].mean(axis=0)) / x[train].std(axis=0).clip(.01))
                     @ rng.normal(size=(x.shape[1], 200)) / np.sqrt(x.shape[1]))
    legacy = min((ridge(hidden[train], y[train], alpha) for alpha in penalties(hidden[train])),
                 key=lambda m: np.mean((hidden[val] @ m[0] + m[1] - y[val])**2))
    legacy_prediction = hidden[test] @ legacy[0] + legacy[1]
    fits = {}
    best = {}
    smoothers = {s: onset_smoother(s) for s in (0., .4, 3.)} if erp else {0.: None}
    for width in (.5, 2., 8.):
        kernels, params = zip(*(block_kernel(x, train, width) for x in blocks), strict=True)
        for mode, mixture in MIXTURES.items():
            kernel = sum(w*k for w, k in zip(mixture, kernels, strict=True) if w)
            # Profile regularization in the centered Gram eigensystem.
            col_mean = kernel[train].mean(axis=0)
            centered = kernel - kernel.mean(axis=1, keepdims=True) - col_mean + col_mean.mean()
            eigenvalues, vectors = np.linalg.eigh(centered[train])
            coordinates = vectors.T @ (y[train] - reference)
            for alpha in (.01, .1, 1., 10.):
                dual = vectors @ (coordinates / (np.maximum(0., eigenvalues[:, None]) + alpha))
                mapped = centered[val] @ dual
                for smooth, operator in smoothers.items():
                    estimate = apply_smoother(mapped, operator) + reference
                    loss = float(np.mean((estimate - y[val])**2))
                    if mode not in best or loss < best[mode][0]:
                        best[mode] = (loss, dict(
                            width=width, alpha=alpha, smoothing=smooth, mixture=mixture,
                            blocks=params, kernel_mean=col_mean, dual=dual, offset=reference
                        ), centered)
    for mode, (loss, model, centered) in best.items():
        # Absorb both Gram centering and onset regularization into the exported fit.
        dual = apply_smoother(model["dual"], smoothers[model["smoothing"]])
        dual = dual - dual.mean(axis=0)
        model["dual"] = dual
        model["offset"] = reference - model.pop("kernel_mean") @ dual
        pred = centered[test] @ dual + reference
        fits[mode] = dict(model=model, validation_mse=loss, prediction=pred)
    modes = ("hybrid_light", "hybrid_state", "hybrid", "hybrid_more_state")
    selected = min(modes, key=lambda mode: fits[mode]["validation_mse"])
    deploy = min(fits, key=lambda mode: fits[mode]["validation_mse"])
    measurement = {"legacy_nonlinear": scores(y[test], legacy_prediction, reference, seed)}
    for mode, info in fits.items():
        ratings = scores(y[test], info["prediction"], reference, seed)
        paired = np.mean((legacy_prediction - y[test])**2 - (info["prediction"] - y[test])**2, axis=1)
        draws = rng.integers(len(test), size=(1000, len(test)))
        ratings.update(paired_mse_reduction=float(paired.mean()),
                       paired_ci95=np.quantile(paired[draws].mean(axis=1), [.025, .975]).tolist(),
                       **{key: info["model"][key] for key in ("width", "alpha", "smoothing")})
        if erp:
            error = (info["prediction"] - y[test]).reshape(len(test), -1, ERP_BINS)
            ratings["rmse_by_onset_bin"] = np.sqrt(np.mean(error**2, axis=(0, 1))).tolist()
        measurement[mode] = ratings
    measurement["selected_hybrid"] = selected
    measurement["selected_calibration"] = deploy
    return measurement, fits[selected if require_state else deploy], slopes, (train, val, test), blocks


def apply_smoother(values, operator):
    if operator is None:
        return values
    return (values.reshape(len(values), -1, ERP_BINS) @ operator).reshape(values.shape)
