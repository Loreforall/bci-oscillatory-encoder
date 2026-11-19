"""Compare the browser recording with counterfactual images and optional THINGS-EEG2."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

import cv2
import numpy as np

from .things import CHANNELS, verify_release


def summary(epochs, rate):
    n = rate // 2
    segments = np.lib.stride_tricks.sliding_window_view(epochs, n, axis=-1)[..., ::n//2, :]
    window = np.hanning(n)
    power = abs(np.fft.rfft((segments-segments.mean(axis=-1,keepdims=True))*window))**2
    hz = np.fft.rfftfreq(n, 1/rate)
    mean_power = power.reshape(-1, len(hz)).mean(axis=0)
    bands = [(2, 8), (8, 14), (14, 30), (30, 46), (46, 86)]
    band_power = np.array([mean_power[(hz >= a) & (hz < b)].sum() for a,b in bands])
    centered = epochs - epochs.mean(axis=-1, keepdims=True)
    normalized = centered / np.maximum(1e-12, np.linalg.norm(centered, axis=-1, keepdims=True))
    matrix = normalized @ np.swapaxes(normalized, -1, -2)
    upper = np.triu_indices(matrix.shape[-1], 1)
    sd = np.sqrt(np.mean(centered**2, axis=-1))
    return dict(relative_band_power=(band_power / band_power.sum()).tolist(),
                median_absolute_channel_correlation=float(np.median(np.abs(matrix[..., upper[0], upper[1]]))),
                median_temporal_sd=float(np.median(sd)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('research-data/things-eeg2'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figure', type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as folder, zipfile.ZipFile(args.root/'test_images.zip') as archive:
        source = next(p for p in archive.namelist() if '_aircraft_carrier/' in p and p.endswith('.jpg'))
        pixels = cv2.imdecode(np.frombuffer(archive.read(source), dtype=np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(cv2.resize(pixels, (64,64), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        alternate = Path(folder)/'alternate.rgb'
        alternate.write_bytes(rgb.tobytes())
        binary = Path(folder)/'recordings.f64'
        subprocess.run(['node', str(Path(__file__).with_name('recording_signals.mjs')), str(alternate), str(binary)], check=True)
        primary, seed, gray, other = np.fromfile(binary, dtype='<f8').reshape(4,1440,256)
    rms = lambda x: float(np.sqrt(np.mean(x**2)))
    report = dict(alternate_condition=source.split('/')[-2], rate_hz=480, baseline_s=.5,
                  same_seed_baselines_equal=bool(np.array_equal(primary[:240], gray[:240])
                                                 and np.array_equal(primary[:240],other[:240])),
                  stimulus_rms=rms(primary[240:]),
                  stimulus_difference_vs_gray=rms(primary[240:]-gray[240:]),
                  stimulus_difference_vs_other=rms(primary[240:]-other[240:]),
                  seed_difference=rms(primary-seed),
                  final_half_second_rms=rms(primary[-240:]),
                  maximum_absolute_sample=float(np.abs(primary).max()))
    selected = np.round(np.linspace(0,255,8)).astype(int)
    matched = primary[240:600,selected].T
    recording, *_ = verify_release(args.root,'01')
    channels = [recording['ch_names'].index(ch) for ch in CHANNELS]
    eeg = recording['preprocessed_eeg_data'][:16,:8][:,:,channels,20:170]
    report['bands_hz'] = [[2,8],[8,14],[14,30],[30,46],[46,86]]
    report['simulated_proxy_750ms'] = summary(matched[None,...],480)
    report['measured_eeg_750ms'] = summary(eeg,200)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if args.figure:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        time=np.arange(1440)/480-.5
        fig, (one,many) = plt.subplots(2,1,figsize=(11,8),height_ratios=[1,4],sharex=True,layout='constrained')
        for name,data in [('Neighborhood',primary),('Gray',gray),('Aircraft carrier',other)]:
            one.plot(time,data[:,119],lw=.75,label=name)
        one.legend(loc='upper right')
        one.set_ylabel('S120 · a.u.')
        for row,j in enumerate(np.round(np.linspace(0,255,16)).astype(int)):
            many.plot(time,primary[:,j]/max(1e-12,3*primary[:,j].std())-row,color='#273949',lw=.6)
        many.set_yticks(-np.arange(16),[f'S{j+1}' for j in np.round(np.linspace(0,255,16)).astype(int)])
        many.set_xlabel('Time relative to onset (s)')
        for ax in (one,many):
            ax.axvline(0,color='#9a4c24',lw=.85)
            ax.grid(axis='x',alpha=.2)
        fig.savefig(args.figure,dpi=144)


if __name__=='__main__':
    main()
