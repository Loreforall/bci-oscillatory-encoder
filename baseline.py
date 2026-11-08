"""Deterministic visual-feature to oscillatory-template mapping.

A computational baseline only: NOT an estimate of actual neural responses.
"""
import numpy as np

BAND_FREQUENCIES = (6.0, 10.0, 20.0, 40.0)
FEATURE_NAMES = ('brightness', 'contrast', 'edge_density', 'low_frequency',
                 'high_frequency', 'foveal', 'peripheral', 'motion')


def encode_features(features, channels=8):
    """Return power [T,C,4], phase [T,C,4], spatial [T,C]."""
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 16 or x.shape[0] == 0:
        raise ValueError('Expected a nonempty [frames,16] feature array')
    if not np.isfinite(x).all() or channels < 1:
        raise ValueError('Features must be finite and channels positive')
    brightness, contrast, edges, low, high, foveal, peripheral, motion = [np.clip(x[:, i], 0, 1) for i in range(8)]
    base = np.stack([0.15 + 0.6 * motion, 0.15 + 0.6 * brightness,
                     0.15 + 0.6 * contrast, 0.15 + 0.6 * (edges + high) / 2], axis=-1)
    channel_scale = np.linspace(0.8, 1.2, channels, dtype=np.float32)
    power = base[:, None, :] * channel_scale[None, :, None]
    # Phase is a declared deterministic convention, not a neural phase prediction.
    phase_base = np.pi * (2 * np.stack([motion, brightness, contrast, edges], axis=-1) - 1)
    phase = np.broadcast_to(phase_base[:, None, :], power.shape).copy()
    center = np.linspace(-1, 1, channels, dtype=np.float32)
    preference = foveal[:, None] * (1 - np.abs(center))[None, :] + peripheral[:, None] * np.abs(center)[None, :] + 0.05
    spatial = preference / preference.sum(axis=1, keepdims=True)
    return {'band_power': power.astype(np.float32), 'band_phase': phase.astype(np.float32),
            'spatial_weights': spatial.astype(np.float32)}
