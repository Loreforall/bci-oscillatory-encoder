# COVE research methodology and contribution assessment

## A. Research overview

COVE is a computational demonstration of the visual encoding methodology proposed in the author's two supplied abstracts. The first, **Encoding the World into Brain Frequencies: A Pathway for Immersive BCIs**, proposes a broader information-encoding framework. It starts with sensory and motor frequency signatures, maps real-world dynamics into organized oscillatory patterns, and treats individual calibration and precise timing as requirements for a future bidirectional system. The second, **Encoding the World into the Brain Frequencies: A Neural Vision Framework**, specializes that idea to vision. It specifies Fourier and wavelet decomposition across central and peripheral vision, structured cross-frequency and phase-amplitude modulation, and a future feedback path using visual evoked responses and attention signals.

Both abstracts envisage delivery and perceptual effects as a larger research objective. The repository demonstrates the computational vision pathway and observational learning from measured EEG. The claims about sensory delivery, selective entrainment and perception without retinal input remain hypotheses. The repository also preserves a simulated adaptation experiment for studying latency and robustness. Its signals and simulations are in software, and the project makes no inference about perceptual effects from regression performance. This account uses the full abstracts supplied in the author's earlier research-portfolio attachment and the final engineering assignment.

## B. Proposed methodology mapped to the implementation

| Proposed concept | Formulation and implementation | Demonstration |
| --- | --- | --- |
| Encode a scene into organized oscillatory patterns | Complex field $z_{ik}(t)$, 25 visual sites × four modes, driven by spatially resolved image features. [Python encoder](cove.py) and [browser encoder](../frontend/lib/cove.ts). | Structured waveform, state trajectory, spatial maps and JSON field export. |
| Fourier spatial-frequency decomposition | Radial RMS of the mean-removed 2D FFT in four bins. [Extractors](cove.py) and [shared browser FFT](../frontend/lib/features.ts). | Fourier band visualization and quantitative features, also available to the detailed calibration kernel. |
| Wavelet temporal-frequency analysis | Causal undecimated Haar differences at 1, 2 and 4 frame half-windows. | Temporal feature maps. Signed Haar coefficients enter the complex drive for video. |
| Foveated and peripheral representation | Exponential radial retinotopic sampling, eccentricity-dependent attenuation of two oriented quadrature banks, and local divisive contrast normalization. | Actual normalized site coordinates, central/peripheral luminance, contrast and orientation drive maps. |
| Frequency, amplitude, and phase | Complex analytic coordinates, mode-specific rotation, bounded forcing, and zero-initialized dynamics. | Component-level state, magnitude, wrapped phase and temporal spectrum. |
| Spatial and cross-frequency coupling | Contrast-dependent four-neighbor spatial mixing with self-edges. The low mode's imaginary coordinate shifts higher-mode angular rates and its real coordinate gates their drive. | Incoming edge weights, drive gain, frequency shift, relative phase, and spatial disagreement energy. |
| Individual calibration against measured responses | Training-only masked spatial slope fit and participant-specific Gaussian kernel ridge readout on visual and optional oscillatory blocks. | Predicted post-onset ERP in eight labeled posterior channels with the model's held-out metrics. |
| Inverse visual-to-oscillatory encoding | Learned affine renderer from field coordinates, followed by a bounded code update against known visual pixels with graph and temporal penalties. | Code-only decode and input-guided refinement with separate RMSE. |
| Closed-loop feedback | Earlier pilot's bounded simulated readout adaptation with imposed delay and noise. [Simulation and results](README.md#results). | Offline numerical study. Live evoked or attention feedback remains future work. |

The frequency pathway is the organizing method. Reconstruction checks how much visual structure the field carries. EEG prediction separately checks whether a calibrated readout generalizes to held-out stimuli. The strongest ERP readout may use the visual bypass alone while the complex field remains fully generated and inspectable.

## C. Mathematical specification and audit

[Advanced COVE equations and derivations](ADVANCED.md#visual-operators) give the image operators, field recurrence, proof of contraction, calibrated readout and inverse objective. The browser and Python use equivalent real/imaginary arithmetic, including post-step recording at the integration clock. Phase is circular. The ordering of indices, amplitude threshold and update timing are defined in the [export specification](EXPORTS.md).

For a fixed contrast field let $A$ be the symmetric positive edge affinities, $q_i=\sum_j A_{ij}$ and $W_{ij}=A_{ij}/q_i$. The adimensional graph energy exported for mode $k$ is

$$
E_{\mathrm{space},k}
=\frac12\sum_{ij} A_{ij}|z_{ik}-z_{jk}|^2
= z_k^*(\operatorname{diag}(q)-A)z_k\ge0.
$$

This equality follows by expanding the squared differences and using symmetry. On each row $A_{ii}=1$, so $A_{ij}=W_{ij}/W_{ii}$. The exported value measures instantaneous spatial disagreement; the driven time-varying recurrence makes no monotonic energy claim.

For frozen low-mode coordinates, the local generator for the phase substep decomposes into $\sum\omega_k|Z_{ik}|^2$ and $2.4\sum k\,\operatorname{Im}(z_{i0})|Z_{ik}|^2$. Holding the coefficients fixed gives a norm-preserving rotation. The spatial term above is the corresponding graph disagreement diagnostic. These constructions explain what the split classical operators do. The recurrence includes damping, graph averaging and external drive, and has no single conserved Hamiltonian. An additional field-theory dynamical term was unnecessary. The useful mathematical guarantee is a strictly contractive same-input update on the complex polydisc.

## D. Implementation and resource use

The Python research implementation in [cove.py](cove.py), [hybrid.py](hybrid.py), and [decoder.py](decoder.py) builds the visual, field and evaluation paths. The browser drives the same operations from decoded RGB frames through [cove.ts](../frontend/lib/cove.ts). CSS and the original deterministic encoder, plots, and exports remain in the existing workbench. [The exporter](../frontend/lib/cove-export.ts) serializes the advanced state independently of chart selection.

For processed side $N$, spatial FFT work scales as $O(FN^2\log N)$ over $F$ frames. Direct browser quadrature sampling scales as $O(FSKJ^2)$ for $S$ sites, $K$ orientations and a $J$-pixel kernel side, with bounded Gaussian passes and bilinear interpolation. The sparse dynamical update takes $O(TK(S+E))$ for $T$ internal steps and $E$ graph edges. Full complex history uses $16TSK$ bytes for double-precision components. The browser's chart bins may average adjacent spectrogram windows on long clips and the field export preserves the numerical sequence. The decoder solves only the displayed window of at most eight frames. Gaussian kernel training profiles a centered Gram eigendecomposition per candidate width and mixture, scaling cubically in the training-condition count. Exported inference is linear in the number of training anchors at fixed feature dimensions.

RGB preview retains decoded color; the advanced dynamics use luminance at 64 × 64. Fourier energy features discard spatial phase and enter only the optional detailed readout. The drive instead uses signed local quadratures, site luminance, motion and wavelet detail. The 12 × 12 grayscale renderer intentionally limits spatial resolution. These losses are explicit in the design. [Schema and units](EXPORTS.md) give array dimensions and coordinate conventions.

## E. Experimental validation

The [original pilot report](README.md#data-and-alignment) documents THINGS-EEG2 provenance, licenses, SHA-256 checks and condition-level alignment, including the limits of repeat-to-event identification. The final engineering pass reran both participants, three concept-held-out splits with 120 train, 40 validation and 40 test conditions, and both the historical nonlinear baseline and current visual-only and hybrid kernels. The [measured results and ablations](ADVANCED.md#onset-aware-eeg-readout) are preserved.

- ERP MSE skill against the training mean was 8.09% and 10.47% for the previous nonlinear visual baseline, 10.76% and 13.01% for the visual-only kernel, and 10.75% and 12.95% for the validation-selected hybrid. A reproducible incremental benefit from adding oscillatory state to the strongest visual-only pathway was absent in this two-participant pilot. Log-power results were inconsistent.
- Code-only decoding and known-input refinement gave RMSE of 0.2301 and 0.1203 on the 200 natural images, and 0.1621 and 0.0679 on 24 unseen procedural video clips. Procedural-video frame-difference RMSE fell from 0.0256 to 0.0121. The reconstruction experiment does not use measured EEG.
- A same-input stability test decreased maximum complex-state separation from 1.4142 to 0.0349 over eight frames. The regression suite checks the contraction bound and checks Python against TypeScript at every integration step of a test site, as well as parity for visual features, endpoint states and inverse outputs.

The advanced EEG JSON report, decoder weights and decoder metrics were reproduced byte-for-byte. See [commands](ADVANCED.md#reproduce-the-implementation-and-evidence).

## F. Computational demonstration

1. Select an image sequence, a video or the built-in moving grating and click **Run**. The existing visual preview, baseline results and exports remain available.
2. Open **Inspect distributed COVE**. Choose a frame, site, mode and Haar half-window. Inspect the retinotopy, feature decomposition, field amplitude and phase, site signal and spectra, spatial weights and cross-mode diagnostics.
3. Compare the forward reconstruction with the input-guided refinement. The latter uses original downsampled visual pixels. Neither output represents a person's recorded percept.
4. For a single uploaded still, optionally load a local ERP calibration generated with the command in [Advanced COVE](ADVANCED.md). The reported test skill is a calibration-dataset metric, distinct from the estimated time trace for the uploaded still.
5. Use **Export COVE JSON** for every integration step and the frame-level feature, graph and phase metadata. Use **Export COVE CSV** for tabular frame-level states. Files are generated locally.

Browser verification used an actual uploaded PNG, a VP8 WebM with 18 decoded samples at 12 fps, the built-in 48-frame sequence, and an exported sub-01 hybrid calibration on the still. Browser downloads were parsed from disk. The still exported a complex array of shape $25\times25\times4\times2$, 100 CSV state rows, and a separately labeled 240-value ERP prediction. Browser media sampling and codec availability depend on the browser. Continuous-video EEG prediction has no validated calibration in the pilot.

## G. Research contribution assessment

- **Direct implementation of the proposed methodology.** The two abstracts motivate decomposition of visual events into frequency structure, foveated/peripheral encoding, and spatially and temporally organized oscillatory templates with cross-frequency interactions and individual calibration. The demonstration realizes this processing chain on images and video.
- **Established techniques.** FFTs, Haar wavelets, Gabor quadrature, radial warps, row-normalized local graphs, complex rotation, ridge regression, Gaussian kernels and bootstrap scoring are established building blocks.
- **Potentially distinctive constructions.** The engineered combination maps causal wavelet detail into oriented, foveated complex drives, couples sites through a reversible contrast-conditioned graph, and uses the low mode's two quadratures to govern higher-mode drive and phase separately. The phase/amplitude split admits an explicit contraction margin. A parallel visual pathway preserves information the field compresses. These are precise architectural distinctions; priority or independent novelty would require a literature and prior-art comparison.
- **Demonstrated contributions.** Inspectable and exportable complex fields, cross-runtime numerical agreement, bounded stable trajectories, visual reconstruction with measured precision, and modest observational ERP prediction on held-out stimuli.
- **Unvalidated hypotheses.** Biological interpretation of individual mode phases, causal effects of the encoded field, perception, real-time physiological feedback, generalization across participants or clinical use remain outside the demonstrated evidence.

## H. Final summary

COVE produces structured, locally inspectable spatiotemporal oscillatory patterns and exposes the numerical field and its coupling relationships. It decodes and refines visual reconstructions, preserves the legacy workbench, and supports participant-specific observational ERP estimates on still images. The strongest EEG result supports a useful visual readout on this stimulus dataset. It leaves the broader biological and perceptual hypotheses open.
