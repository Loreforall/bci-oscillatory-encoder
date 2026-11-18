"""Export small Python reference values for advanced browser parity."""
import json
from pathlib import Path

import numpy as np

from research.cove import coupling, encode, extract, real_code
from research.decoder import refine


def main():
    root = Path(__file__).resolve().parents[1]
    y, x = np.mgrid[:64, :64]
    frames = np.asarray([np.stack(((13*x + 7*y + 17*t) % 256,
                                   (3*x + 11*y + 7*t) % 256,
                                   (5*x + 2*y + 29*t) % 256), axis=-1)
                         for t in range(5)], dtype=np.uint8)
    visual = extract(frames, 24)
    graph = coupling(visual.contrast, (2., 0.))
    samples = []
    code = real_code(encode(visual, advanced=True, capture=samples.append))
    trace = np.asarray(samples)[:, 12]
    trace = np.stack([trace.real, trace.imag], axis=-1)
    renderer = json.loads((root / "frontend/lib/cove-decoder.json").read_text())
    refined = refine(code, visual.pixels, graph, renderer)
    result = dict(site_12_trace=trace.ravel().tolist(), spectrum=visual.spectrum.tolist(), haar=visual.wavelets[4, 12].tolist(),
                  code=code[4].tolist(), feature=visual.conventional[4].tolist(),
                  detail=visual.detail[4].tolist(), pixels=visual.pixels[4].tolist(),
                  reconstruction=(refined[4] @ np.asarray(renderer["weights"]) +
                                  renderer["offset"]).tolist())
    (root / "frontend/tests/advanced-reference.json").write_text(
        json.dumps(result, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
