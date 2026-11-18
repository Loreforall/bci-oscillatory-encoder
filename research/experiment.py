"""Sequence/stimulus-disjoint benchmarks and target-free inverse refinement."""
import numpy as np

from .cove import GRID, FREQUENCIES, Visual, coupling, encode, fit_coupling, real_code


def penalties(x):
    centered = x - x.mean(axis=0)
    energy = max(1e-12, np.sum(centered**2) / x.shape[1])
    return energy * np.array([.001, .01, .1, 1., 10., 100., 1000.])


def ridge(x, y, alpha):
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    centered = x - x_mean
    weights = np.linalg.solve(centered.T @ centered + alpha * np.eye(x.shape[1]),
                              centered.T @ (y - y_mean))
    return weights, y_mean - x_mean @ weights


def split(groups, seed):
    unique = np.random.default_rng(seed).permutation(np.unique(groups))
    if len(unique) < 10:
        raise ValueError("At least ten independent stimulus groups are required")
    train, val, test = np.split(unique, [int(.6 * len(unique)), int(.8 * len(unique))])
    return tuple(np.flatnonzero(np.isin(groups, part)) for part in (train, val, test))


def scores(y, predicted, reference, seed):
    error = ((y - predicted)**2).reshape(len(y), -1).mean(axis=1)
    baseline = ((y - reference)**2).reshape(len(y), -1).mean(axis=1)
    draw = np.random.default_rng(seed).integers(len(y), size=(1000, len(y)))
    ratio = error[draw].mean(axis=1) / np.maximum(1e-30, baseline[draw].mean(axis=1))
    return {"rmse": float(np.sqrt(error.mean())),
            "skill": float(1 - error.mean() / max(1e-30, baseline.mean())),
            "skill_ci95": np.quantile(1 - ratio, [.025, .975]).tolist()}


def inverse(state, pixels, decoder, neural, graph, *, radius=.15, steps=35):
    """Refine against known visual input and preserve reference model predictions."""
    weights, offset = decoder
    neural_weights, _ = neural
    anchor = state.copy()
    refined = state.copy()
    visual_scale = 1 / pixels.shape[1]
    neural_scale = .15 / neural_weights.shape[1]
    difference = np.eye(GRID**2) - graph
    laplacian = difference.transpose(0, 2, 1) @ difference
    graph_scale = .035
    rate = 1 / (2 * visual_scale * np.linalg.norm(weights, 2)**2
                + 2 * neural_scale * np.linalg.norm(neural_weights, 2)**2 + .28 + 2 * graph_scale * np.linalg.norm(laplacian, 2, axis=(1, 2)).max())
    for _ in range(steps):
        shift = refined - anchor
        temporal = np.diff(shift, axis=0)
        gradient = (2 * visual_scale * (refined @ weights + offset - pixels) @ weights.T
                    + 2 * neural_scale * (shift @ neural_weights) @ neural_weights.T
                    + .12 * shift
                    + 2 * graph_scale * (laplacian @ shift.reshape(len(state), GRID**2, -1)).reshape(state.shape))
        gradient[:-1] -= .04 * temporal
        gradient[1:] += .04 * temporal
        update = (shift - rate * gradient).reshape(len(state), -1, 2, len(FREQUENCIES))
        # Per-mode trust region and unit disk, projected by scaling the displacement.
        update /= np.maximum(1, np.linalg.norm(update, axis=2, keepdims=True) / radius)
        origins = anchor.reshape(update.shape)
        candidate = origins + update
        outside = np.linalg.norm(candidate, axis=2) > 1
        a = (update**2).sum(axis=2)
        b = (origins * update).sum(axis=2)
        c = (origins**2).sum(axis=2) - 1
        fraction = np.divide(-b + np.sqrt(np.maximum(0, b*b - a*c)), a,
                             out=np.ones_like(a), where=outside)
        candidate = origins + update * fraction[:, :, None, :]
        refined = candidate.reshape(state.shape)
    return refined


def feedback(x, y, clean_targets, model, delay, rate=.12):
    """Prequential bounded calibration with delayed noisy observations and fixed offset."""
    weights, offset = model
    shift = np.zeros_like(weights)
    adaptive = []
    for time, current in enumerate(x):
        due = time - delay - 1
        if due >= 0:
            past = x[due]
            error = y[due] - (past @ (weights + shift) + offset)
            shift *= .998
            shift += rate * np.outer(past, error) / (.1 + past @ past)
            shift /= max(1, np.linalg.norm(shift) / 4)
        adaptive.append(current @ (weights + shift) + offset)
    return {"adaptive_rmse": float(np.sqrt(np.mean((np.array(adaptive) - clean_targets)**2))),
            "fixed_rmse": float(np.sqrt(np.mean((x @ weights + offset - clean_targets)**2))),
            "final_update_norm": float(np.linalg.norm(shift))}


def design(visuals, parameters, fps, mode):
    if mode == "pixels":
        return [visual.pixels for visual in visuals]
    if mode in ("visual", "learned", "supervised"):
        return [visual.conventional for visual in visuals]
    codes = []
    for visual in visuals:
        static = len(visual.drive) == 1
        if static:
            visual = Visual(np.repeat(visual.drive, 5, axis=0),
                            np.repeat(visual.contrast, 5, axis=0),
                            visual.conventional, visual.pixels)
        code = real_code(encode(visual, parameters if mode != "fixed" else (2., 0.), fps,
                                phase=mode != "no_phase", spatial=mode != "no_spatial",
                                cross=mode != "no_cross"))
        codes.append(code[-1:] if static else code)
    return codes


def benchmark(visuals, responses, groups, seed, fps):
    train, val, test = split(groups, seed)
    if len(np.unique(np.asarray(groups)[test])) != len(test):
        raise ValueError("Aggregate repeats before scoring independent stimulus groups")
    parameters = fit_coupling([visuals[index] for index in train])
    y = np.asarray(responses)
    y_train = np.concatenate(y[train])
    reference = y_train.mean(axis=0)
    selected_alpha = {}
    results = {}
    predictions = {}
    rng = np.random.default_rng(seed)
    projection = rng.normal(size=(len(visuals[0].conventional[0]), 2 * GRID**2 * len(FREQUENCIES)))
    for mode in ("pixels", "pca", "visual", "learned", "supervised", "fixed", "no_phase",
                 "no_spatial", "no_cross", "cove"):
        x = np.asarray(design(visuals, parameters, fps, "visual" if mode == "pca" else mode))
        if mode == "pca":
            flat = np.concatenate(x[train])
            mean = flat.mean(axis=0)
            _, _, components = np.linalg.svd(flat - mean, full_matrices=False)
            x = (x - mean) @ components[:32].T
        if mode == "supervised":
            flat = np.concatenate(x[train])
            mean = flat.mean(axis=0)
            target_scale = y_train.std(axis=0).clip(1e-8)
            cross_covariance = (flat - mean).T @ ((y_train - reference) / target_scale)
            directions, _, _ = np.linalg.svd(cross_covariance, full_matrices=False)
            x = (x - mean) @ directions[:, :32]
        if mode == "learned":
            # Nonlinear random-feature map with a learned readout and shared alpha grid.
            flat = np.concatenate(x[train])
            x = np.tanh(((x - flat.mean(axis=0)) / flat.std(axis=0).clip(.01))
                        @ projection / np.sqrt(projection.shape[0]))
        x_train = np.concatenate(x[train])
        candidates = [(alpha, ridge(x_train, y_train, alpha)) for alpha in penalties(x_train)]
        selected_alpha[mode], (weights, offset) = min(candidates, key=lambda item: np.mean(
            (x[val] @ item[1][0] + item[1][1] - y[val])**2))
        prediction = x[test] @ weights + offset
        predictions[mode] = prediction
        results[mode] = scores(y[test], prediction, reference, seed)
    pixels = np.array([visual.pixels for visual in visuals])
    decoder = ridge(x_train, np.concatenate(pixels[train]), .1)
    refined = np.array([inverse(x[i], pixels[i], decoder, (weights, offset),
                                coupling(visuals[i].contrast, parameters)) for i in test])
    results["inverse"] = {
        "reconstruction_rmse": float(np.sqrt(np.mean(
            (refined @ decoder[0] + decoder[1] - pixels[test])**2))),
        "initial_reconstruction_rmse": float(np.sqrt(np.mean(
            (x[test] @ decoder[0] + decoder[1] - pixels[test])**2))),
        "response": scores(y[test], refined @ weights + offset, reference, seed),
    }
    paired = ((y[test] - predictions["visual"])**2 -
              (y[test] - prediction)**2).reshape(len(test), -1).mean(axis=1)
    draws = rng.integers(len(test), size=(1000, len(test)))
    results["cove_minus_visual_mse_reduction"] = {
        "mean": float(paired.mean()),
        "ci95": np.quantile(paired[draws].mean(axis=1), [.025, .975]).tolist(),
    }
    results["selected_alpha"] = selected_alpha
    return results, (x, weights, offset, test, prediction), parameters
