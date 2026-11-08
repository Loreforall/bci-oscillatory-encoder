# Visual-to-Oscillatory Research Workbench

Experimental software that extracts visual features from short videos and maps them to **synthetic** multiband oscillatory templates. This is a computational research demonstration, **not** a neural stimulation system, validated visual-cortex encoder, or medical device.

## Run locally

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

## Private-core deployment

Deploy the Gradio app as a Python service (e.g. Render, Railway or a private Hugging Face Space). A hosted Gradio interface does not expose source via the UI, but **hosting a public GitHub repository or a public Space makes the source publicly accessible**. Use a **private repository** and a hosting provider that supports deployment from it. Set resource limits, restrict uploads, and monitor service usage before sharing widely.

## Data and safety

No stimulation waveforms, tACS/TMS/tFUS parameters, device-control instructions or claims of brain entrainment are provided. Generated waveforms are visualization-only, in arbitrary units. Avoid uploading sensitive video. The app writes transient export files to the host's temporary directory; configure cleanup and retention before public launch.

## Intellectual property

Copyright remains with its owner. No open-source license is granted by this repository's absence of a LICENSE file. Review dependency licenses, contributions and any patent strategy before distributing source or technical implementation details. A public demo does not itself grant software licensing rights.
