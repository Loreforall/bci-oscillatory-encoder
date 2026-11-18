"""Render aggregate scientific figures from the committed JSON results."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("research/results"))
    args = parser.parse_args()
    directory = args.directory
    real = json.loads((directory / "things.json").read_text())["results"]
    synthetic = json.loads((directory / "synthetic.json").read_text())["results"]

    models = ("pixels", "pca", "visual", "learned", "supervised", "fixed",
              "no_phase", "no_spatial", "no_cross", "cove")
    labels = ("Pixel ridge", "Visual PCA", "Visual ridge", "Nonlinear visual",
              "Supervised visual", "Fixed coupling", "No phase", "No spatial",
              "No cross-mode", "Full COVE")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharey=True, layout="constrained")
    for ax, task, title in zip(axes, ("erp", "log_power"), ("ERP", "Log band power"), strict=True):
        for offset, (subject, info), color in zip((-.13, .13), real.items(),
                                                 ("#176F91", "#A3482A"), strict=True):
            run = info["runs"][0]["tasks"][task]["results"]
            point = np.array([run[model]["skill"] for model in models])
            lo, hi = np.array([run[model]["skill_ci95"] for model in models]).T
            ax.errorbar(point, np.arange(len(models)) + offset,
                        xerr=[np.maximum(0, point-lo), np.maximum(0, hi-point)],
                        fmt="o", color=color, capsize=2, ms=4, label=subject)
        ax.axvline(0, color="#40474C", lw=1, ls="--")
        ax.set_title(title, loc="left")
        ax.set_xlabel("MSE skill vs. training mean")
        ax.set_yticks(range(len(models)), labels)
        ax.grid(axis="x", alpha=.2)
    axes[0].invert_yaxis()
    axes[1].legend(frameon=False)
    fig.suptitle("Held-out THINGS-EEG2 image conditions · seed 17 · 95% condition bootstrap")
    fig.savefig(directory / "held-out-eeg.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    ax = axes[0]
    for subject, info in real.items():
        times = info["alignment"]["erp_bin_centers_ms"]
        runs = [r["tasks"]["erp"]["cove_rmse_by_bin"] for r in info["runs"]]
        ax.plot(times, np.mean(runs, axis=0), label=subject)
    ax.set(xlabel="Time after image onset (ms)", ylabel="RMSE (scaled CSD)", title="COVE ERP error by time")
    ax.legend(frameon=False)
    ax = axes[1]
    for noise in (0, .08, .25):
        matching = [[r for r in run["feedback"] if r["noise_sd"] == noise] for run in synthetic]
        latency = [1000*r["delay_frames"]/24 for r in matching[0]]
        ax.plot(latency, np.mean([[r["adaptive_rmse"] for r in run] for run in matching], axis=0),
                marker="o", label=f"Adaptive σ={noise:g}")
    ax.axhline(np.mean([r["feedback"][0]["fixed_rmse"] for r in synthetic]),
               ls="--", color="black", lw=1, label="Fixed")
    ax.set(xlabel="Added feedback delay (ms)", ylabel="RMSE (synthetic units)",
           title="Prequential simulated feedback")
    ax.legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(directory / "erp-and-feedback.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
