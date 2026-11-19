"""Compare exploratory cortical features on the frozen stimulus partitions."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from .cove import coupling, fit_coupling
from .experiment import scores, split
from .hybrid import MIXTURES, block_kernel, components, onset_smoother
from .things import CHANNELS, load


def response_features(visuals):
    frames = []
    for visual in visuals:
        contrast = visual.contrast[0]
        graph = coupling(visual.contrast, [2, 2])[0]
        frames.append(dict(detail=visual.detail[0].tolist(), contrast=contrast.tolist(),
                           motion=visual.conventional[0].reshape(len(contrast), -1)[:, 2].tolist(),
                           weights=[[dict(j=int(j), weight=float(row[j])) for j in np.flatnonzero(row)]
                                    for row in graph]))
    runner = Path(__file__).with_name("cortical_features.mjs")
    result = subprocess.run(["node", str(runner)], input=json.dumps(dict(frames=frames)),
                            capture_output=True, check=True, text=True)
    return np.asarray(json.loads(result.stdout))


def gram_fit(kernels, y, train, val, test):
    best = (np.inf, None, None)
    offset = y[train].mean(axis=0)
    smoothers = [(s, onset_smoother(s)) for s in (0., .4, 3.)]
    for label, kernel in kernels:
        col = kernel[train].mean(axis=0)
        centered = kernel - kernel.mean(axis=1, keepdims=True) - col + col.mean()
        eigenvalues, vectors = np.linalg.eigh(centered[train])
        y_projected = vectors.T @ (y[train] - offset)
        for alpha in (.01, .1, 1., 10.):
            coefficients = vectors @ (y_projected / (np.maximum(eigenvalues[:, None], 0) + alpha))
            val_pred = centered[val] @ coefficients
            for weight, smoother in smoothers:
                predicted = (val_pred.reshape(len(val), -1, 30) @ smoother).reshape(len(val), -1) + offset
                loss = np.mean((y[val] - predicted)**2)
                if loss < best[0]:
                    test_pred = centered[test] @ coefficients
                    best = (loss, (test_pred.reshape(len(test), -1, 30) @ smoother).reshape(len(test), -1)
                            + offset, dict(kernel=label, alpha=alpha, smoothing=weight))
    return best[1:]


def compare(visuals, cortical, target, groups, seed):
    train, val, test = split(groups, seed)
    original = components(visuals, fit_coupling([visuals[j] for j in train]))
    y = target.reshape(len(groups), -1)
    options = {key: [] for key in ("linear_visual", "original_cove", "visual_only", "existing_hybrid",
                                    "cortical_only", "hybrid_plus_cortical")}
    x = original[0]
    normalized = (x - x[train].mean(axis=0)) / x[train].std(axis=0).clip(.01)
    options["linear_visual"].append(("linear", normalized @ normalized[train].T / x.shape[1]))
    for width in (.5, 2., 8.):
        visual, detail, cove, cortex = [block_kernel(block, train, width)[0] for block in (*original, cortical)]
        for key, block in (("original_cove", cove), ("visual_only", visual), ("cortical_only", cortex)):
            options[key].append((str(width), block))
        for name in ("hybrid_light", "hybrid_state", "hybrid", "hybrid_more_state"):
            a,b,c = MIXTURES[name]
            hybrid = a*visual + b*detail + c*cove
            options["existing_hybrid"].append((f"{name}/{width}", hybrid))
            options["hybrid_plus_cortical"].append((f"{name}/{width}", .9*hybrid + .1*cortex))
    ratings, predictions = {}, {}
    for key, kernels in options.items():
        pred, parameters = gram_fit(kernels, y, train, val, test)
        ratings[key] = {**scores(y[test], pred, y[train].mean(axis=0), seed), **parameters}
        predictions[key] = pred
    gain = ((y[test] - predictions["existing_hybrid"])**2
            - (y[test] - predictions["hybrid_plus_cortical"])**2).mean(axis=1)
    draws = np.random.default_rng(seed).integers(len(test), size=(1000, len(test)))
    ratings["paired_gain_vs_hybrid"] = {"mse_reduction": float(gain.mean()),
                                      "ci95": np.quantile(gain[draws].mean(axis=1), [.025,.975]).tolist()}
    return ratings, dict(schema="cove-measured-erp-v1", condition=int(test[0]+1), seed=seed,
                         channels=list(CHANNELS), times_ms=list(range(10,600,20)),
                         measured=target[test[0]].tolist(),
                         cortical_readout=predictions["hybrid_plus_cortical"][0].reshape(target.shape[1:]).tolist())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("research-data/things-eeg2"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, help="Optional local-only measured and predicted held-out ERP JSON")
    args=parser.parse_args()
    report={}
    for subject in ("01","02"):
        visuals, groups, erp, _, _ = load(args.root, subject)
        cortical=response_features(visuals)
        report[subject]=[]
        for seed in (17,29,43):
            ratings, comparison=compare(visuals,cortical,erp,groups,seed)
            report[subject].append(dict(seed=seed, **ratings))
            if args.comparison and subject=="01" and seed==17:
                args.comparison.write_text(json.dumps(dict(subject=subject, **comparison))+"\n")
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__=="__main__":
    main()
