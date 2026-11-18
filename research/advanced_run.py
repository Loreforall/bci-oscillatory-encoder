"""Evaluate the hybrid pilot with the existing concept partitions."""
import argparse
import json
from pathlib import Path

from .hybrid import fit
from .things import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("research-data/things-eeg2"))
    parser.add_argument("--out", type=Path, default=Path("research/results/hybrid.json"))
    args = parser.parse_args()
    report = {}
    for subject in ("01", "02"):
        visuals, groups, erp, power, _ = load(args.root, subject)
        report[subject] = [
            {"seed": seed, **{label: fit(visuals, target, groups, seed, erp=(label == "erp"))[0]
                             for label, target in (("erp", erp), ("power", power))}}
            for seed in (17, 29, 43)
        ]
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
