import torch


class WaveformSynthesizer:
    def __init__(self, sampling_rate=250, duration=1.0):
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.band_frequencies = torch.tensor([6.0, 10.0, 20.0, 40.0])
        if sampling_rate <= 80 or duration <= 0:
            raise ValueError("Sampling rate must exceed 80 Hz and duration must be positive")

    def synthesize(self, template):
        power = template["band_power"]
        phase = template["band_phase"]
        spatial = template["spatial_weights"]

        device = power.device
        band_freqs = self.band_frequencies.to(device)

        batch, time_steps, channels, bands = power.shape

        t = torch.arange(int(self.sampling_rate * self.duration), device=device,
                         dtype=power.dtype) / self.sampling_rate

        waveform = torch.zeros(
            batch,
            time_steps,
            channels,
            len(t),
            device=device
        )

        if bands != len(band_freqs):
            raise ValueError("Expected four frequency bands")
        for b in range(bands):
            freq = band_freqs[b]
            amp = power[:, :, :, b].unsqueeze(-1)
            ph = phase[:, :, :, b].unsqueeze(-1)

            wave = amp * torch.sin(2 * torch.pi * freq * t + ph)
            waveform += wave

        spatial = spatial.unsqueeze(-1)
        waveform = waveform * spatial

        # Do not clip: clipping introduces harmonics not present in the template.

        return waveform
