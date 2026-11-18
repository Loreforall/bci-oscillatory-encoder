# COVE numerical export specification

Use **Inspect distributed COVE** after a run. These files describe computational population states in arbitrary units. The optional ERP record contains observational predictions in the THINGS-EEG2 derivative's scaled CSD units.

## JSON

The schema identifier is `cove-field-v1`, and `signal_type` is `modeled_population_field`.

| Variable | Shape or interpretation |
| --- | --- |
| `source` | Input provenance from the legacy run. `kind` identifies images, video or synthetic; `width` and `height` are decoded input dimensions in pixels, `frames` counts processed frames, `fps` is the requested input sampling frequency, and `sampling` identifies the browser procedure. Videos also record `duration_seconds`. `processed_frames_sha256` identifies the concatenated processed RGB bytes. |
| `input_frame_count`, `model_frame_rate_hz`, `integration_rate_hz`, `steps_per_frame`, `internal_first_time_s` | Counts and clocks of the integration. A multi-frame clip uses the requested sampling frequency and $\lceil100/f_s\rceil$ steps per input frame. A single sample is held for five intervals at 24 Hz. The first stored sample is after one integration step; initialization is an all-zero complex field. |
| `coordinates` | 25 row-major site records with zero-based `site`, `row`, `column`; x increases to the right, y downward. `x_normalized` and `y_normalized` are normalized to the processed image range [−1,1]. `eccentricity` is radial normalized image distance from the center. Sites approximate a foveated image grid. |
| `nominal_frequencies_hz` | Four mode centers in Hz. |
| `spatial_frequency_edges_cycles_per_pixel` | Three radial Fourier bin boundaries in processed image cycles/pixel. |
| `haar_half_windows_frames` | Three lengths of each half of a Haar comparison window in input frames. |
| `spatial_slopes` | Two nonnegative coefficients on normalized position L1 distance and absolute luminance-contrast difference. |
| `internal_complex_shape`, `internal_complex_flat` | Shape $[T,25,4,2]$ and row-major flat array of doubles in order time, site, mode, component. Components are real and imaginary, in arbitrary units. Sample $j$ has time $(j+1)/f_{\rm integration}$. Compute magnitude with hypot and phase with atan2; near-zero magnitudes have undefined display phase. |
| `frames` | One record per input sample. Fields are listed below. |
| `observational_erp` | Null or `subject` release alias, the eight ordered `channels`, 30 `onset_bin_centers_ms`, and `prediction_scaled_csd_flat` in channel-major order. `held_out` is null for legacy calibration or contains RMSE in scaled CSD, fractional skill, bootstrap intervals, paired MSE reductions in squared scaled CSD, width/alpha/smoothing model parameters, and per-onset-bin RMSE. None of these fields are field samples. |

Each frame record has the following variables.

| Variable | Shape or interpretation |
| --- | --- |
| `frame`, `input_sample_time_s`, `state_time_s` | Zero-based frame index, nominal source sampling time in seconds, and time of the state at the end of the hold. Browser seeks can return repeated or skipped source frames. |
| `fourier_band_rms` | Four non-DC luminance RMS values from low to high spatial frequency, in normalized luminance units. |
| `haar_detail` | $[25,3]$ signed temporal detail in normalized luminance units. |
| `luminance`, `contrast`, `motion_proxy` | 25 site values each. Luminance and local standard deviation use normalized grayscale; signed motion is bounded by the tanh scaling. |
| `conventional_features` | 375 site-major values, per site in order luminance, contrast, motion, four normalized quadrature phase-change proxies, four low-scale and four high-scale quadrature magnitudes. The input frame rate scales the signed motion proxy. |
| `quadrature_features` | 400 signed normalized outputs, site-major then four orientations then low real, low imaginary, high real, high imaginary. Both quadrature arrays are in filter-response units. |
| `pixels_12x12` | 144 row-major grayscale means on [0,1]. These describe the processed input, not decoded pixels. |
| `complex_drive` | $[25,4,2]$ bounded complex input per site and mode, before the spatial input mixture and gate. |
| `states` | $[25,4]$ records with real `re`, imaginary `im`, `amplitude` and `phase_rad`. Phase is wrapped to [−π,π] and null at amplitude ≤ $10^{-9}$. These states are the last integration sample of each frame. |
| `spatial_weights` | 25 sparse receiver lists of `source_site` and dimensionless `weight`, indexed as $W_{\rm receiver,source}$. Each receiver sums to one and includes itself. Weights are fixed within the corresponding input hold. |
| `cross_mode` | $[25,4]$ records. `source_mode` is zero for higher modes or null for the lowest; `frequency_shift_hz` is a signed angular-rate increment converted to Hz; `drive_gain` multiplies forcing; `relative_phase_rad` is the wrapped phase of target times conjugate of source and null if either amplitude is too small or the mode has no source. Coefficients are evaluated from the endpoint state. To reproduce the applied coefficients at each step, evaluate the same operator on the preceding full-rate sample. |
| `spatial_dirichlet_energy` | Four nonnegative graph quadratic forms, in squared modeled state units. Formula and interpretation are in [the methodology audit](METHODOLOGY.md#c-mathematical-specification-and-audit). |

## CSV

There is one row per frame, site and mode, ordered by those zero-based indexes. Timestamp columns refer to the input sampling and modeled state times separately. Site geometry repeats as row, column, normalized x and y, and eccentricity. Each row carries nominal Hz, complex real and imaginary parts, magnitude, wrapped phase, the source mode, frequency shift in Hz, dimensionless drive gain and wrapped phase difference. Unavailable phases and source indices are empty cells. The JSON is the authoritative export for sparse graphs, raw substeps, features and the observational ERP record.

## Read the field in Python

```python
import json
import numpy as np

with open("cove-field.json") as stream:
    report = json.load(stream)
data = np.asarray(report["internal_complex_flat"]).reshape(report["internal_complex_shape"])
field = data[..., 0] + 1j * data[..., 1]
times = np.arange(1, len(field) + 1) / report["integration_rate_hz"]
site_index, mode_index = 12, 1
amplitude = np.abs(field[:, site_index, mode_index])
phase = np.where(amplitude > 1e-9, np.angle(field[:, site_index, mode_index]), np.nan)
```
