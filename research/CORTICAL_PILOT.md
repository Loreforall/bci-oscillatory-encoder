# Exploratory visual cortical response pilot

The COVE reference encodes visual features as a rotating complex field. Its real component is an encoder state, and its existing ERP calibration is a separately fitted prediction. The new pathway models signed deviations of excitatory and inhibitory populations at the same retinotopic sites. The demonstration does not assign individual neurons or EEG electrodes to sites. It runs four orientation-specific populations per site, with two spatial-frequency inputs, contrast-dependent resonance and damping, lateral coupling, delayed stimulus drive and fast and slow adaptation. It is deterministic. The E–I plane magnitude and angle summarize those modeled variables in arbitrary units.

The population-pair approach is inspired by established excitatory and inhibitory population modeling. Its settings are phenomenological and the pathways have no anatomical head model. [Wilson and Cowan (1972)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1484078/)

## Observational check

The pilot used the existing verified THINGS-EEG2 test sets for participants 01 and 02, each with 200 stimulus conditions. It used the frozen concept splits for seeds 17, 29 and 43. There were 120 training, 40 validation and 40 test conditions per split. Readout regularization, kernel width, hybrid choice and temporal smoothing were selected on validation data. The exploratory simulation used the 5 × 5 pilot grid, a 100 ms input, and a 600 ms observation window. The browser defaults to 16 × 16, which was not evaluated against EEG here. [THINGS-EEG2 dataset paper](https://doi.org/10.1016/j.neuroimage.2022.119754)

Mean held-out skill across three splits, relative to the training mean ERP. Higher is better.

| Readout | Participant 01 | Participant 02 |
| --- | ---: | ---: |
| Conventional linear visual ridge | 0.07831 | 0.11055 |
| Original COVE states | 0.00860 | 0.01504 |
| Visual-only kernel | 0.10762 | 0.13007 |
| Existing hybrid | 0.10755 | 0.12954 |
| Cortical features only | 0.08566 | 0.11116 |
| Hybrid with cortical features | 0.10808 | 0.12702 |

The cortical readout improved on the COVE-only readout in this pilot. Adding it to the hybrid did not improve held-out skill consistently. The splits reuse conditions, and the two participants do not support population-level conclusions. The readout predicts trial-averaged, processed EEG in scaled CSD units. This is neither scalp source localization nor validation of the model's local population trajectories. Per-split paired uncertainty and readout parameters are in [the result file](results/cortical.json). The prior benchmark is unchanged.

## Reproduce or inspect a held-out comparison

Use the existing verified local dataset and Node.js 24 or later. The optional comparison JSON contains one genuine measured trial average and its calibrated prediction. Keep that file local and select it under **Measured EEG comparison** on the site. Both series use the same 20 ms post-onset bins. The comparison stimulus is independent of the demonstration photograph and video.

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m research.cortical_run \
  --out /tmp/cove-cortical.json \
  --comparison /tmp/cove-heldout-erp.json
```
