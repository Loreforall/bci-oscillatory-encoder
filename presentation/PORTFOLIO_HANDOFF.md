# COVE · Visual-to-Oscillatory Encoding

**Short description.** A computational research project and interactive workbench that translate images and short videos into spatially organized, time-varying oscillatory patterns. The demonstration lets visitors follow the transformation and inspect the underlying numerical output.

## Research motivation

A visual scene changes across space and time. COVE asks how to represent those changes as a structured field with interpretable location, frequency, amplitude and phase while retaining useful visual information. The immediate objective is a working computational encoder with inspectable outputs, an evaluated visual reconstruction path and an observational comparison with measured EEG.

The project builds on two research abstracts. *Encoding the World into Brain Frequencies: A Pathway for Immersive BCIs* frames the broader problem of encoding sensory events into oscillatory representations for future bidirectional interfaces. *Encoding the World into the Brain Frequencies: A Neural Vision Framework* specifies the visual pathway, including spatial and temporal frequency analysis, foveated and peripheral processing, and interactions between oscillatory modes. COVE implements and tests the computational vision pathway. The perceptual and physiological hypotheses in the abstracts remain open.

## Method and implemented capabilities

Images and videos are analyzed for spatial frequency, temporal change, orientation and contrast. A retinotopic approximation samples both central and peripheral regions. These local features drive interacting complex modes at 25 modeled sites. The sequence retains frequency, phase and amplitude through time, with contrast-dependent spatial interactions and modulation between modes. A parallel visual-feature pathway retains information useful for observational response prediction.

The project includes Python and browser implementations, numerical stability analysis, cross-runtime verification, CSV and JSON exports, a decoder from oscillatory coordinates, and an optional input-guided refinement. A participant-specific readout estimates still-image event-related potentials (ERPs) when a calibration is provided. Relevant technical areas include Fourier and wavelet analysis, complex-valued dynamical systems, foveated image processing, sparse graph operators, regularized inverse problems, and Gaussian kernel regression.

## Verified results and scope

| Pilot measure | Result |
| --- | --- |
| Held-out ERP MSE skill for two THINGS-EEG2 participants | Hybrid 10.75% / 12.95%. Previous nonlinear visual baseline 8.09% / 10.47%. Strongest visual-only kernel 10.76% / 13.01%. |
| Natural-image grayscale reconstruction RMSE | 0.2301 from the forward code; 0.1203 after refinement using known input pixels. |
| Procedural-video grayscale reconstruction RMSE | 0.1621 from the forward code; 0.0679 after input-guided refinement. |

ERP skill is the reduction in mean-squared error relative to the training-mean predictor, averaged over three concept-held-out splits of this two-participant pilot. The visual-only comparison gives comparable results, leaving the incremental predictive value of the oscillatory representation unresolved. Reconstruction concerns low-resolution grayscale visual information. The refinement uses known pixels, and neither metric measures human perception. The pilot supports observational still-image prediction. Continuous-video EEG calibration, live feedback, and perceptual effects remain unvalidated.

## Demonstration and suggested presentation

Visitors can run the built-in sequence or upload an image or video, select a retinotopic site and mode, and play or scrub the computed trajectory. The source image, waveform cursor and activity maps share the same timing. Advanced analysis exposes spectral and wavelet summaries, spatial and cross-frequency interactions, reconstructions, optional ERP traces and numerical exports.

- **Overview.** One sentence on the research question and a workbench screenshot showing the input, selected-mode waveform and retinotopic activity.
- **Method.** A high-level flow diagram from visual input to feature decomposition to complex field, with separate branches for visual reconstruction and optional EEG calibration.
- **Interactive workbench.** A short capture of time scrubbing and comparing central and peripheral sites. Label all field imagery as modeled output.
- **Validation.** A compact held-out ERP comparison against both visual baselines and a forward/refined reconstruction pair labeled with the refinement's input access.
- **Research scope.** A short statement on the two-participant pilot, evidence boundaries and reproducibility.

## Contribution and portfolio summary

Suggested attribution: “Proposed and developed a visual-to-oscillatory methodology combining foveated spatial and temporal features with a stable, interacting complex field, and evaluated it through a reproducible software workbench.” Fourier analysis, wavelets, complex rotation and kernel regression are established building blocks. The research contribution under assessment is their particular architectural integration, computational interpretation and evaluation.

**Concise technical summary.** COVE is a reproducible visual-to-oscillatory encoder for images and video. It builds a spatial complex field from foveated features and exposes its computed dynamics in an interactive browser workbench. Separate experiments evaluate visual reconstruction and participant-specific still-image ERP prediction on held-out THINGS-EEG2 stimuli.
