"""Export an exploratory, participant-specific still-image ERP readout."""
import argparse
import json

import numpy as np
from pathlib import Path


from .experiment import benchmark, ridge, split
from .things import CHANNELS, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("research-data/things-eeg2"))
    parser.add_argument("--subject", choices=("01", "02"), default="01")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    visuals, groups, erp, _, _ = load(args.root, args.subject)
    targets = erp.reshape(len(erp), 1, -1)
    results, (x, weights, offset, _, _), coupling = benchmark(visuals, targets, groups, 17, 24)
    train, _, _ = split(groups, 17)
    pixels = np.concatenate([visuals[index].pixels for index in train])
    decoder, decoder_offset = ridge(np.concatenate(x[train]), pixels, .1)
    # Export the evaluated training-only fit. Keep its test condition data out of the model.
    model = dict(schema="cove-erp-v1", subject=f"sub-{args.subject}", channels=CHANNELS,
                 coupling=coupling, weights=weights.tolist(), offset=offset.tolist(),
                 decoder=decoder.tolist(), decoder_offset=decoder_offset.tolist(),
                 calibration_seed=17, fit_conditions=len(train),
                 validation_alpha=results["selected_alpha"]["cove"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(model, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
