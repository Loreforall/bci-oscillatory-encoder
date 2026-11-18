"""Generate a compact browser parity fixture from the NumPy implementation."""
import json
from pathlib import Path

import numpy as np

from research.cove import Visual, coupling, encode, extract, real_code


def main():
    y, x = np.mgrid[:64, :64]
    frames = np.array([(120 + 80 * np.cos(.4*x + .32*y + phase)).astype(np.uint8)
                       for phase in (0., .7, 1.4)])
    video = extract(frames, fps=13)
    still = extract(frames[:1])
    repeated = Visual(np.repeat(still.drive, 5, axis=0), np.repeat(still.contrast, 5, axis=0),
                      still.conventional, still.pixels)
    slopes = (1., 3.)
    output = dict(video_code=real_code(encode(video, slopes, 13)).tolist(),
                  selected_weights=coupling(video.contrast, slopes)[:, 12].tolist(),
                  still_code=real_code(encode(repeated, slopes, 24))[-1].tolist())
    path = Path(__file__).resolve().parents[1] / "frontend/tests/cove-reference.json"
    path.write_text(json.dumps(output, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
