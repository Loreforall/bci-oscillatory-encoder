# COVE · CV research notes

**Research project title.** COVE — Visual-to-Oscillatory Encoding.

**Focus.** Computational representation of visual input as distributed oscillatory patterns with interpretable time, location, frequency, amplitude and phase. Secondary evaluations address visual information preservation and observational EEG prediction.

**Connection to the research abstracts.** The project develops the computational vision pathway described in *Encoding the World into Brain Frequencies: A Pathway for Immersive BCIs* and *Encoding the World into the Brain Frequencies: A Neural Vision Framework*. The first frames sensory encoding for a future bidirectional interface. The second motivates Fourier and wavelet analysis, foveal and peripheral processing, and structured mode interactions for vision.

## Facts available for tailoring

- Proposed a visual-to-oscillatory architecture that integrates foveated visual features, a spatial complex field, interactions between modes, and participant-specific calibration. An auxiliary visual pathway preserves features for readout.
- Implemented image and video feature extraction, complex state evolution, adaptive spatial mixing, and regularized inverse reconstruction in Python and TypeScript.
- Used Fourier transforms, causal Haar wavelets, oriented quadrature filtering, retinotopic sampling, complex-valued dynamical systems, sparse graph operators, ridge and Gaussian kernel regression, and bootstrap evaluation.
- Delivered a Next.js and React research workbench with synchronized computed playback, retinotopic maps, spectra, reconstructions, optional ERP prediction, and JSON/CSV exports.
- Evaluated two authentic THINGS-EEG2 participant datasets across three concept-held-out splits. Hybrid ERP MSE skill relative to the training-mean predictor was 10.75% and 12.95%. The earlier nonlinear visual comparator achieved 8.09% and 10.47%; a stronger visual-only kernel achieved 10.76% and 13.01%.
- Evaluated code-only and input-guided grayscale reconstruction separately. Natural-image RMSE was 0.2301 and 0.1203, respectively. Procedural-video RMSE was 0.1621 and 0.0679.
- Documented numerical stability, ran cross-runtime parity tests, and reproduced the benchmark and decoder results byte-for-byte in the reference environment.

**Potentially relevant skills.** Scientific Python, NumPy, OpenCV, TypeScript, React, Next.js, digital signal processing, numerical modeling, EEG data alignment, experimental design, inverse optimization, and reproducible software testing.

## Example CV bullets

- Developed COVE, a visual-to-oscillatory encoder with a stable, spatially distributed complex state and an inspectable browser workbench for images and video.
- Evaluated participant-specific ERP prediction on concept-held-out THINGS-EEG2 images, achieving 10.75% and 12.95% MSE skill against the training-mean predictor in a two-participant pilot.
- Implemented numerical exports, cross-runtime tests, and repeatable benchmarks for state dynamics, decoding and input-guided reconstruction.

**Scope to retain in CV wording.** The stronger visual-only kernel was comparable to the hybrid; independent predictive benefit from oscillatory features remains unresolved. The work is computational. Avoid claims of demonstrated stimulation, induced perception, clinical validity, established priority, publication, patent status, or affiliations without separate evidence.
