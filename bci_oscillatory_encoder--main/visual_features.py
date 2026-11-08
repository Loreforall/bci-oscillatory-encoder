import cv2
import numpy as np


class VisualFeatureExtractor:
    def __init__(self, image_size=(64, 64)):
        self.image_size = image_size

    def preprocess_frame(self, frame):
        frame = cv2.resize(frame, self.image_size)
        frame = frame.astype(np.float32) / 255.0
        return frame

    def brightness(self, gray):
        return np.mean(gray)

    def contrast(self, gray):
        return np.std(gray)

    def edge_density(self, gray):
        edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
        return np.mean(edges > 0)

    def spatial_frequency_fft(self, gray):
        # Relative non-DC spectral energy; both outputs are in [0, 1].
        centered = gray.astype(np.float64) - np.mean(gray)
        spectrum = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
        h, w = spectrum.shape
        yy, xx = np.ogrid[:h, :w]
        radius = np.sqrt(((yy - h // 2) / max(h / 2, 1)) ** 2 +
                         ((xx - w // 2) / max(w / 2, 1)) ** 2)
        total = float(spectrum.sum())
        if total < 1e-12:
            return 0.0, 0.0
        low = float(spectrum[(radius > 0) & (radius <= 0.35)].sum() / total)
        high = float(spectrum[radius > 0.35].sum() / total)
        return low, high

    def foveal_peripheral_activity(self, gray):
        h, w = gray.shape
        ch, cw = h // 2, w // 2

        foveal = gray[
            ch - h // 6:ch + h // 6,
            cw - w // 6:cw + w // 6
        ]

        mask = np.ones_like(gray, dtype=bool)
        mask[
            ch - h // 6:ch + h // 6,
            cw - w // 6:cw + w // 6
        ] = False

        peripheral = gray[mask]

        return np.mean(foveal), np.mean(peripheral)

    def motion_intensity(self, gray, previous_gray):
        if previous_gray is None:
            return 0.0

        diff = np.abs(gray - previous_gray)
        return np.mean(diff)

    def extract_sequence_features(self, frames):
        if not frames:
            raise ValueError("At least one RGB frame is required")
        features = []
        previous_gray = None

        for frame in frames:
            frame = self.preprocess_frame(frame)

            gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gray = gray.astype(np.float32) / 255.0

            bright = self.brightness(gray)
            cont = self.contrast(gray)
            edge = self.edge_density(gray)
            low_freq, high_freq = self.spatial_frequency_fft(gray)
            foveal, peripheral = self.foveal_peripheral_activity(gray)
            motion = self.motion_intensity(gray, previous_gray)

            feature_vector = [
                bright,
                cont,
                edge,
                low_freq,
                high_freq,
                foveal,
                peripheral,
                motion
            ]

            features.append(feature_vector)
            previous_gray = gray

        features = np.array(features, dtype=np.float32)

        wavelet_features = self.wavelet_temporal_features(features)

        full_features = np.concatenate([features, wavelet_features], axis=1)

        return full_features

    def wavelet_temporal_features(self, features):
        wavelet_outputs = []

        for i in range(features.shape[1]):
            signal = features[:, i]

            # A single-frame sequence has no temporal variation.
            if len(signal) == 1:
                wavelet_outputs.append(np.zeros_like(signal))
                continue
            responses = []
            for scale in (1, 2, 3):
                t = np.arange(-4 * scale, 4 * scale + 1, dtype=np.float64) / scale
                kernel = np.cos(5 * t) * np.exp(-t * t / 2)
                kernel -= kernel.mean()
                kernel /= max(np.sum(np.abs(kernel)), 1e-12)
                padded = np.pad(signal, (len(kernel)//2,), mode="edge")
                responses.append(np.convolve(padded, kernel, mode="valid"))
            coeff_mean = np.mean(np.abs(responses), axis=0)

            wavelet_outputs.append(coeff_mean)

        wavelet_outputs = np.stack(wavelet_outputs, axis=1)

        return wavelet_outputs
