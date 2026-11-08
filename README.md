# Visual-to-Oscillatory Research Workbench

Experimental software that extracts visual features from short videos and maps them to **synthetic** multiband oscillatory templates. This is a computational research demonstration, **not** a neural stimulation system, validated visual-cortex encoder, or medical device.

## Web workbench

The public interface uses Next.js and TypeScript. FastAPI calls the original Python feature extractor, deterministic encoder, optional GRU, and waveform synthesizer. Use the synthetic example, a short video, or a sequence of still images to inspect computed signals and export the run.

See [Workbench deployment](DEPLOYMENT.md) for local setup, Vercel settings, backend configuration, and production verification.

- `frontend/` contains the public workspace, methods, and collaboration information.
- `backend/` exposes the bounded encoding API.
- `Dockerfile` packages the private CPU service.
- The original research modules and Gradio application remain at the repository root.

## Run the original Gradio application locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open the local address printed by Gradio. Upload a short video, view the four-band template and synthesized waveform, then export `.npz` results. The public demonstration uses the deterministic baseline and does **not** require trained weights.

## Train and evaluate the research GRU

```bash
python train.py --epochs 80 --output artifacts/model.pt
python compare.py --checkpoint artifacts/model.pt
python -m pytest -q
```

The GRU is trained to reproduce deterministic synthetic teacher targets, not recorded EEG or visual percepts. Comparison with a constant predictor is a *software-learning sanity check*, not a claim of neural encoding performance.

## Data and safety

No stimulation waveforms, tACS/TMS/tFUS parameters, device-control instructions or claims of brain entrainment are provided. Generated waveforms are visualization-only, in arbitrary units. Avoid uploading sensitive video. The original Gradio app writes transient export files to the host's temporary directory; configure cleanup and retention before public launch.

## Intellectual property

Copyright remains with its owner. No open-source license is granted by this repository's absence of a LICENSE file. Review dependency licenses, contributions and any patent strategy before distributing source or technical implementation details. A public demo does not itself grant software licensing rights.
