# v1.0 development snapshot

This is a computational demonstration, not a validated brain-computer interface or neural stimulation system.

## Included
- Original GRU model and training prototype retained for audit.
- Fixed non-DC spatial FFT features and single-frame handling and a dependency-free multiscale Morlet-style temporal filter.
- Added deterministic 16-feature to four-band oscillatory baseline with explicit assumptions.
- Fixed sample grid and removed waveform clipping that introduced harmonics.
- Added Gradio video demonstration and downloadable NPZ results.
- Added core unit tests.

## Not yet established
- No biological validation, measured EEG targets or reliable neural-response predictions.
- Original GRU training targets include random phase and spatial weights; do not use its untrained output as scientific evidence.
- No public deployment, external evaluation, patent assessment or license agreement has occurred.

## Run locally
`pip install -r requirements.txt`
`python -m unittest discover -s tests -v`
`python app.py`

## Distribution
Keep the source repository private pending an IP and dependency-license review. Public access to a hosted demo does not grant permission to copy the implementation. No license is granted by this snapshot.
