"""Portable reconstruction from oscillatory coordinates and held-out measurements."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .cove import FREQUENCIES, GRID, coupling, encode, extract, real_code
from .experiment import design, ridge
from .things import load


def procedural(seed, count):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:64, :64] / 64
    sequences = []
    for _ in range(count):
        frames = []
        texture = rng.normal(size=(8, 8))
        texture = cv2.resize(texture, (64, 64), interpolation=cv2.INTER_CUBIC)
        angle, phase, frequency, drift = rng.uniform(0, np.pi), rng.uniform(0, 6), rng.uniform(2, 12), rng.uniform(-.4, .4)
        u = xx * np.cos(angle) + yy * np.sin(angle)
        bias, center, size = rng.uniform(.3, .7), rng.uniform(.25, .75, 2), rng.uniform(.015, .18)
        for t in range(8):
            envelope = np.exp(-((xx-center[0]-.012*t)**2 + (yy-center[1])**2)/size)
            picture = bias + (.2*np.cos(2*np.pi*frequency*u + phase + t*drift)
                              + .14*texture) * envelope + .08*texture
            frames.append(np.rint(255 * np.clip(picture, 0, 1)).astype(np.uint8))
        sequences.append(extract(np.asarray(frames)))
    return sequences


def model(visuals, slopes):
    x = np.concatenate(design(visuals, slopes, 24, "advanced"))
    y = np.concatenate([v.pixels for v in visuals])
    weights, offset = ridge(x, y, .012)
    weights = weights.round(7)
    # Bound the curvature with the spectral norm of the rounded renderer.
    rate = 1 / (2 * np.linalg.norm(weights, 2)**2 / y.shape[1] + .28 + .07 * (1+np.sqrt(5))**2)
    return dict(weights=weights, offset=offset.round(7), rate=rate)


def refine(code, pixels, graph, renderer, steps=28):
    """Radial feasible updates in the intersection of two complex disks per mode."""
    weights, offset, rate = (np.asarray(renderer[k]) for k in ("weights", "offset", "rate"))
    z = code.copy()
    delta_graph = np.eye(GRID**2) - graph
    laplacian = delta_graph.transpose(0, 2, 1) @ delta_graph
    shape = (len(code), GRID**2, 2, len(FREQUENCIES))
    for _ in range(steps):
        change = z - code
        derivative = (2 / pixels.shape[-1] * (z @ weights + offset - pixels) @ weights.T
                      + .12 * change
                      + .07 * (laplacian @ change.reshape(len(code), GRID**2, -1)).reshape(code.shape))
        diff = np.diff(change, axis=0)
        derivative[:-1] -= .04 * diff
        derivative[1:] += .04 * diff
        shift = (change - rate * derivative).reshape(shape)
        shift /= np.maximum(1, np.linalg.norm(shift, axis=2, keepdims=True) / .15)
        anchor = code.reshape(shape)
        outside = np.linalg.norm(anchor + shift, axis=2) > 1
        a = (shift**2).sum(axis=2)
        b = (anchor * shift).sum(axis=2)
        c = (anchor**2).sum(axis=2) - 1
        step = np.divide(-b + np.sqrt(np.maximum(0, b*b - a*c)), a,
                         out=np.ones_like(a), where=outside)
        z = (anchor + shift * step[:, :, None, :]).reshape(code.shape)
    return z


def evaluate(visuals, renderer, slopes):
    codes = design(visuals, slopes, 24, "advanced")
    errors, peaks, temporal = [], [], []
    for v, code in zip(visuals, codes, strict=True):
        updated = refine(code, v.pixels, coupling(v.contrast, slopes), renderer)
        base = code @ renderer["weights"] + renderer["offset"] - v.pixels
        final = updated @ renderer["weights"] + renderer["offset"] - v.pixels
        errors.extend(np.stack([np.mean(base**2, axis=1), np.mean(final**2, axis=1)], axis=1))
        if len(code) > 1:
            temporal.extend(np.stack([np.mean(np.diff(base, axis=0)**2, axis=1),
                                      np.mean(np.diff(final, axis=0)**2, axis=1)], axis=1))
        peaks.append(np.max(np.linalg.norm(updated.reshape(-1, GRID**2, 2, len(FREQUENCIES)), axis=2)))
    rmse = np.sqrt(np.mean(errors, axis=0))
    return dict(initial_rmse=float(rmse[0]), refined_rmse=float(rmse[1]),
                refined_psnr_db=float(-20*np.log10(rmse[1])), peak_state_amplitude=float(max(peaks)),
                temporal_difference_rmse=np.sqrt(np.mean(temporal, axis=0)).tolist() if temporal else [])


def stability():
    visual = procedural(161, 1)[0]
    initial = np.full((GRID**2, len(FREQUENCIES)), .5+.5j)
    other = -initial
    x = encode(visual, advanced=True, initial=initial)
    y = encode(visual, advanced=True, initial=other)
    distance = np.max(abs(x-y), axis=(1, 2))
    return dict(initial_distance=float(np.max(abs(initial-other))),
                distance_by_frame=distance.tolist(), peak_amplitude=float(max(abs(x).max(), abs(y).max())))


def write(path, value, *, pretty=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=lambda x: x.tolist(), allow_nan=False,
                               indent=2 if pretty else None, separators=None if pretty else (",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=Path("frontend/lib/cove-decoder.json"))
    parser.add_argument("--report", type=Path, default=Path("research/results/decoder.json"))
    parser.add_argument("--root", type=Path, default=Path("research-data/things-eeg2"))
    args = parser.parse_args()
    slopes = (2., 2.)
    renderer = model(procedural(401, 100), slopes)
    natural, _, _, _, _ = load(args.root, "01")
    write(args.asset, renderer)
    write(args.report, dict(synthetic=evaluate(procedural(402, 24), renderer, slopes),
                            natural=evaluate(natural, renderer, slopes), stability=stability()), pretty=True)


if __name__ == "__main__":
    main()
