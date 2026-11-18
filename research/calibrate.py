"""Export an exploratory, participant-specific still-image ERP readout."""
import argparse
import json
from pathlib import Path

import numpy as np

from .decoder import write
from .hybrid import fit
from .experiment import benchmark, ridge, split
from .things import CHANNELS, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("research-data/things-eeg2"))
    parser.add_argument("--subject", choices=("01", "02"), default="01")
    parser.add_argument("--out", type=Path, required=True)
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument("--legacy", action="store_true", help="Export the original COVE ERP model")
    variant.add_argument("--hybrid", action="store_true", help="Require oscillatory features in the ERP readout")
    args = parser.parse_args()
    visuals, groups, erp, _, _ = load(args.root, args.subject)
    if not args.legacy:
        results, selected, slopes, (train, val, test), _ = fit(visuals, erp, groups, 17, require_state=args.hybrid)
        mode = results["selected_hybrid" if args.hybrid else "selected_calibration"]
        payload = selected["model"]
        payload["blocks"] = [block if weight else None for weight, block in
                             zip(payload["mixture"], payload["blocks"], strict=True)]
        write(args.out, dict(schema="cove-erp-v2", subject=f"sub-{args.subject}", channels=CHANNELS,
                             coupling=slopes, calibration_seed=17, fit_conditions=len(train),
                             validation_conditions=len(val), test_conditions=len(test),
                             mode=mode, held_out=results[mode], **payload))
        return
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
