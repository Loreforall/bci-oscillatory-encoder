# COVE · Browser disclosure review

This review describes the repository's current static production build. It does not assess what a particular hosting URL is serving.

| Delivered to the browser | Inspectable information |
| --- | --- |
| Legacy feature and encoder code | Visual feature calculations, deterministic mapping, and synthesized waveform rules. |
| Advanced COVE engine | Retinotopic layout, Fourier and Haar feature calculations, oriented filters, complex state update, coupling structure and constants, and playback sampling. |
| Reconstruction path | Bundled procedural decoder weights and the input-guided refinement algorithm. |
| Calibration support | Calibration file schemas, kernel evaluation, and ERP prediction code. An uploaded calibration stays local to the browser session. Subject-specific weights and anchors are inspectable by whoever receives those calibration files. The application does not bundle the participant calibration files. |
| Plots and numerical exports | Retinotopic coordinates, features, complex trajectories, coupling values, reconstruction diagnostics, and an optional observational ERP trace. These outputs reveal behavior even if the computation runs elsewhere. |

Making the GitHub repository private restricts repository access. A publicly hosted static build still sends its executable browser code and bundled assets to visitors. Production browser source maps are disabled in the current configuration; minified JavaScript remains inspectable. Previously distributed copies can remain available to their recipients. The production build does not include the local THINGS-EEG2 archives, Python benchmark scripts, or methodology Markdown as served pages.

If confidentiality later becomes important, the browser could retain playback, plotting and media controls while a server handles selected operators, such as advanced feature extraction, complex-field evolution, decoder weights and refinement, or participant-specific calibration inference. That would change the local-only processing model. The design would need explicit decisions about media transfer, retention, access control and export resolution. Outputs can still support inference about a hidden operator. Repository visibility, the build architecture and existing export behavior remain as implemented.
