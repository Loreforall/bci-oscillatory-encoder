"""Regenerate the compact reference fixture from the original Python research modules.

From the repository root: python -m tests.generate_browser_reference
"""
import json
from pathlib import Path

import numpy as np
import torch

from baseline import encode_features
from synthesizer import WaveformSynthesizer
from visual_features import VisualFeatureExtractor


def main():
    y, x = np.mgrid[:64, :64]
    frames = [
        np.stack([np.full_like(x, value) for value in (240, 33, 110)], axis=-1),
        np.repeat(np.where(x >= 25, 220, 30)[..., None], 3, axis=2),
        np.repeat(np.where(x + y >= 63, 230, 20)[..., None], 3, axis=2),
        np.stack([(3*x + 5*y) % 256, (11*x + 2*y) % 256, (7*x + 13*y) % 256], axis=-1),
        np.repeat(np.where((x//3 + y//3) % 2, 210, 35)[..., None], 3, axis=2),
    ]
    features = VisualFeatureExtractor().extract_sequence_features([frame.astype(np.uint8) for frame in frames])
    # Include the clipping contract and the single-channel linspace edge case.
    clipped = np.array([[-0.3, 1.2, 0.6, 0.4, 1.4, -0.2, 0.7, 1.1] + [0] * 8], dtype=np.float32)
    cases = []
    for rows, channels, sample_rate in ((features, 3, 101), (clipped, 1, 250), (features[:1], 16, 500)):
        template = encode_features(rows, channels)
        tensors = {key: torch.from_numpy(value).unsqueeze(0) for key, value in template.items()}
        wave = WaveformSynthesizer(sampling_rate=sample_rate, duration=0.5).synthesize(tensors)[0].numpy()
        cases.append(dict(features=rows.tolist(), channels=channels, sample_rate=sample_rate,
                          template={key: value.tolist() for key, value in template.items()},
                          waveform_samples=wave[..., [0, 1, 13, -1]].tolist(),
                          waveform_rms=float(np.sqrt(np.mean(wave**2))),
                          waveform_peak=float(np.max(np.abs(wave)))))
    motion = features[:, 7]
    fixture = dict(cases=cases, motion_hz=np.fft.rfftfreq(len(motion), 1/24).tolist(),
                   motion_power=(abs(np.fft.rfft(motion - motion.mean()))**2 / len(motion)**2).tolist())
    path = Path(__file__).resolve().parents[1] / "frontend/tests/python-reference.json"
    path.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
