"""Run reproducible synthetic or authentic observational experiments."""
import argparse
import json
from pathlib import Path

import numpy as np

from .cove import extract
from .experiment import benchmark, feedback, split
from .things import load


def synthetic(config):
    rng = np.random.default_rng(config["synthetic_generator_seed"])
    yy, xx = np.mgrid[:64, :64] / 64
    visuals, responses = [], []
    for _ in range(config["synthetic_sequences"]):
        angle, direction, scale, speed, phase = rng.uniform(
            [0, 0, 3, .1, 0], [np.pi, 2*np.pi, 9, .8, 2*np.pi])
        time = np.arange(config["synthetic_frames"])
        u = xx * np.cos(angle) + yy * np.sin(angle)
        center_x = .5 + .2 * np.cos(direction) * (time / time.max() - .5)
        center_y = .5 + .2 * np.sin(direction) * (time / time.max() - .5)
        envelope = np.exp(-((xx - center_x[:, None, None])**2 +
                            (yy - center_y[:, None, None])**2) / .12)
        phase_time = time * speed + phase
        carrier = np.cos(2*np.pi * scale * u + phase_time[:, None, None])
        frames = np.rint(255 * (.5 + .43 * carrier * envelope)).astype(np.uint8)
        visuals.append(extract(frames, config["fps"]))
        # Analytic visual parameters with independent observation noise, no COVE teacher.
        response = np.stack([np.cos(phase_time) * np.cos(angle),
                             np.sin(phase_time) * scale / 9,
                             np.cos(2 * phase_time) * np.sin(angle)], axis=-1)
        responses.append(response + rng.normal(0, .04, size=response.shape))
    return visuals, np.arange(len(visuals)), np.array(responses)


def run_synthetic(config):
    visuals, groups, responses = synthetic(config)
    runs = []
    for seed in config["split_seeds"]:
        results, (x, weights, offset, test, _), coupling = benchmark(
            visuals, responses, groups, seed, config["fps"])
        test_x = np.concatenate(x[test])
        drift = np.linspace(0, .4, len(test_x))[:, None]
        clean = np.concatenate(responses[test]) + drift
        rng = np.random.default_rng(seed)
        noise = rng.normal(size=clean.shape)
        calibration = [dict(delay_frames=delay, noise_sd=sd, **feedback(
            test_x, clean + sd * noise, clean, (weights, offset), delay))
            for delay in config["feedback_delays_frames"] for sd in config["feedback_noise_sd"]]
        runs.append(dict(seed=seed, results=results, coupling=coupling, feedback=calibration))
    return runs


def run_real(config, root, subjects):
    participants = {}
    for subject in subjects:
        visuals, groups, erp, spectrum, alignment = load(root, subject)
        runs = []
        for seed in config["split_seeds"]:
            tasks = {}
            for name, target in (("erp", erp), ("log_power", spectrum)):
                y = target.reshape(len(target), 1, -1)
                results, (_, _, _, test, prediction), coupling = benchmark(
                    visuals, y, groups, seed, config["fps"])
                residual = target[test] - prediction.reshape(target[test].shape)
                axis_rmse = np.sqrt(np.mean(residual**2, axis=(0, 1)))
                tasks[name] = dict(results=results, coupling=coupling,
                                   cove_rmse_by_bin=axis_rmse.tolist())
                if name == "erp":
                    train, _, _ = split(groups, seed)
                    reference = target[train].mean(axis=0)
                    window = np.hanning(target.shape[-1])
                    actual = np.fft.rfft((target[test] - reference) * window, axis=-1)[..., 3:8]
                    estimate = np.fft.rfft(
                        (prediction.reshape(target[test].shape) - reference) * window,
                        axis=-1)[..., 3:8]
                    magnitude = abs(actual) * abs(estimate)
                    tasks[name]["residual_erp_phase_cosine"] = float(
                        np.real(actual * estimate.conj()).sum() / max(1e-30, magnitude.sum()))
            runs.append(dict(seed=seed, tasks=tasks))
        participants[f"sub-{subject}"] = dict(alignment=alignment, runs=runs)
    return participants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("synthetic", "things"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--root", type=Path, default=Path("research-data/things-eeg2"))
    parser.add_argument("--subjects", nargs="+", choices=("01", "02"), default=["01"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    results = (run_synthetic(config) if args.experiment == "synthetic"
               else run_real(config, args.root, args.subjects))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(experiment=args.experiment, config=config, results=results),
                                   indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
