"""Visual features and stable, computational complex population dynamics."""
from dataclasses import dataclass

import cv2
import numpy as np

SIZE = 64
GRID = 5
FREQUENCIES = np.array([6., 10., 20., 40.])
axis = np.linspace(-1, 1, GRID)
yy, xx = np.meshgrid(axis, axis, indexing="ij")
radius = np.hypot(xx, yy)
eccentricity = np.expm1(np.log(3) * radius / np.sqrt(2)) / 2
scale = np.divide(eccentricity, radius, out=np.zeros_like(radius), where=radius > 0)
POSITIONS = np.stack([yy * scale, xx * scale], axis=-1).reshape(-1, 2)
DISTANCE = np.abs(POSITIONS[:, None] - POSITIONS[None]).sum(axis=-1)
indices = np.indices((GRID, GRID)).reshape(2, -1).T
HAAR_SCALES = (1, 2, 4)
FREQUENCY_EDGES = (.0625, .125, .25)
FFT_RADIUS = np.hypot(*np.meshgrid(np.fft.fftfreq(SIZE), np.fft.fftfreq(SIZE)))
FFT_BAND = np.digitize(FFT_RADIUS, FREQUENCY_EDGES)
LOCAL = np.abs(indices[:, None] - indices[None]).sum(axis=-1) <= 1


def kernel(angle, frequency, phase):
    values = cv2.getGaborKernel((15, 15), 2.8, angle, 1 / frequency, 1, phase,
                               ktype=cv2.CV_64F)
    values -= values.mean()
    return values / np.linalg.norm(values)


FILTERS = [(kernel(angle, frequency, 0), kernel(angle, frequency, -np.pi / 2))
           for frequency in (.10, .21) for angle in np.arange(4) * np.pi / 4]


@dataclass
class Visual:
    drive: np.ndarray
    contrast: np.ndarray
    conventional: np.ndarray
    pixels: np.ndarray
    spectrum: np.ndarray | None = None
    wavelets: np.ndarray | None = None
    detail: np.ndarray | None = None


def extract(frames, fps=24):
    """Accept one sequence of grayscale or RGB uint8 frames."""
    frames = np.asarray(frames)
    if (frames.ndim not in (3, 4) or not frames.size or frames.dtype != np.uint8
            or (frames.ndim == 4 and frames.shape[-1] != 3)
            or not np.isfinite(fps) or fps <= 0):
        raise ValueError("Expected nonempty uint8 grayscale/RGB sequence and positive FPS")
    coords = ((POSITIONS + 1) * (SIZE - 1) / 2).astype(np.float32)
    mx, my = coords[:, 1, None], coords[:, 0, None]

    def sample(image):
        return cv2.remap(image, mx, my, cv2.INTER_LINEAR).ravel()

    maps, contrasts, conventional, pixels = [], [], [], []
    spectral, luminance_history, detail = [], [], []
    previous = None
    previous_response = None
    for frame in frames:
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_AREA) / 255.
        smooth = cv2.GaussianBlur(gray, (0, 0), 2.0)
        luminance = sample(cv2.GaussianBlur(gray, (0, 0), 1))
        contrast = np.sqrt(np.maximum(0, sample(cv2.GaussianBlur(gray**2, (0, 0), 2)
                                                 - smooth**2)))
        motion = np.zeros_like(luminance) if previous is None else np.tanh(
            (luminance - previous) * fps / 12)
        previous = luminance
        luminance_history.append(luminance)
        response = []
        for index, (real, imag) in enumerate(FILTERS):
            source = gray if index < 4 else gray - smooth
            quadrature = sample(cv2.filter2D(source, -1, real)) + 1j * sample(
                cv2.filter2D(source, -1, imag))
            attenuation = 1 + (1 if index < 4 else 5) * eccentricity.ravel() ** 2
            response.append(quadrature / ((.15 + contrast) * attenuation))
        low, high = np.asarray(response).T.reshape(-1, 2, 4).transpose(1, 0, 2)
        temporal = np.zeros_like(low.real) if previous_response is None else np.imag(
            low * previous_response.conj()) / (.1 + abs(low * previous_response))
        previous_response = low
        source = .5 * low + .25 * high + .15 * (luminance[:, None] - .5)
        source += .1 * (motion[:, None] + 1j * temporal)
        maps.append(source / (1 + abs(source)))
        contrasts.append(contrast)
        conventional.append(np.column_stack([luminance, contrast, motion, temporal,
                                               abs(low), abs(high)]).ravel())
        pixels.append(cv2.resize(gray, (12, 12), interpolation=cv2.INTER_AREA).ravel())
        power = abs(np.fft.fft2(gray - gray.mean()))**2 / SIZE**4
        spectral.append(np.sqrt(np.bincount(FFT_BAND.ravel(), weights=power.ravel(), minlength=4)))
        detail.append(np.stack([low.real, low.imag, high.real, high.imag], axis=-1).ravel())
    return Visual(*(np.asarray(items) for items in (maps, contrasts, conventional, pixels, spectral)),
                  haar_details(np.asarray(luminance_history)), np.asarray(detail))


def haar_details(values):
    """Causal, undecimated Haar details with left-edge extension."""
    frames = np.arange(len(values))[:, None]
    return np.stack([
        (values[np.maximum(0, frames - np.arange(size))].mean(axis=1) -
         values[np.maximum(0, frames - size - np.arange(size))].mean(axis=1)) / np.sqrt(2)
        for size in HAAR_SCALES
    ], axis=-1)


def advanced_drive(visual):
    if visual.wavelets is None:
        return visual.drive
    bands = visual.wavelets
    correction = .25 * np.stack([bands[..., 2], bands[..., 1], bands[..., 0],
                                 bands[..., 0] - bands[..., 1]], axis=-1)
    source = visual.drive + 1j * correction
    return source / np.maximum(1., abs(source))


def coupling(contrast, parameters):
    difference = abs(contrast[..., :, None] - contrast[..., None, :])
    logits = -parameters[0] * DISTANCE - parameters[1] * difference
    weights = np.exp(logits) * LOCAL
    return weights / weights.sum(axis=-1, keepdims=True)


def fit_coupling(visuals, penalty=.0002):
    """Fit shared slopes by predicting each site's masked complex drive."""
    source = np.concatenate([visual.drive for visual in visuals])
    contrast = np.concatenate([visual.contrast for visual in visuals])
    off_diagonal = LOCAL & ~np.eye(len(LOCAL), dtype=bool)
    difference = abs(contrast[:, :, None] - contrast[:, None, :])
    candidates = [(a, b) for a in (0., 2., 5., 10.) for b in (0., 2., 6., 12.)]

    def objective(parameters):
        a, b = parameters
        weights = np.exp(-a * DISTANCE - b * difference) * off_diagonal
        predicted = (weights @ source) / weights.sum(axis=-1, keepdims=True)
        return np.mean(abs(predicted - source) ** 2) + penalty * (a*a + b*b)

    return min(candidates, key=objective)


def encode(visual, parameters=(2., 0.), fps=24, *, phase=True, spatial=True, cross=True,
           advanced=False, hold=1, initial=None, capture=None):
    """Unit-disk dynamics with contractive memory and interacting mode transport."""
    if not np.isfinite(fps) or fps <= 0 or not isinstance(hold, int) or hold < 1:
        raise ValueError("Positive finite FPS and positive integer hold are required")
    source = advanced_drive(visual) if advanced else visual.drive
    source = source if phase else abs(source)
    substeps = int(np.ceil(100 / fps))
    dt = 1 / (fps * substeps)
    rotation = np.exp(2j * np.pi * FREQUENCIES * dt)
    retention = np.exp(-dt / .09)
    weights = coupling(visual.contrast, parameters) if spatial else np.broadcast_to(
        np.eye(len(LOCAL)), (len(source), *LOCAL.shape))
    state = np.zeros_like(source[0], dtype=complex) if initial is None else np.asarray(initial, complex).copy()
    if state.shape != source[0].shape or not np.isfinite(state).all() or np.any(abs(state) > 1):
        raise ValueError("Initial state must have the per-mode shape and lie in the unit disk")
    gates = .18 * (np.arange(len(FREQUENCIES)) > 0) / 2 if advanced and cross else 0.
    states = []
    for drive, weight in zip(source, weights, strict=True):
        mixed_drive = .8 * drive + .2 * (weight @ drive)
        for _ in range(substeps * hold):
            # Low-mode state modulates higher-mode phase within the contraction margin.
            transport = rotation * (np.exp(2.4j * dt * state[:, :1].imag *
                                          np.arange(len(FREQUENCIES))) if cross else 1)
            # Higher-mode forcing is gated by the low-mode in-phase component.
            gate = 1 - gates * (1 - state[:, :1].real)
            state = retention * transport * (weight @ state) + (1 - retention) * gate * mixed_drive
            if capture is not None:
                capture(state.copy())
        states.append(state)
    return np.asarray(states)


def real_code(state):
    return np.concatenate([state.real, state.imag], axis=-1).reshape(len(state), -1)
