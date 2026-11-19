# Spatial resolution and visualization verification

The original 5 × 5 reference data remain the basis of the EEG evaluation, trained reconstruction and Advanced Analysis exports. The homepage defaults to 16 × 16 and can also compute 10 × 10 or 5 × 5. Each grid applies the existing foveated transform to its own regular input lattice. It samples luminance, contrast and the orientation filters at those new coordinates before integrating a separate complex state at every site. The input raster is still 64 × 64 and the filter widths are unchanged. Denser sampling therefore increases spatial coverage while nearby samples can still overlap. The grids are computational representations of visual coordinates.

The self edge and orthogonal neighbor rule retain the same degree bound and row normalization at every resolution. Consequently the fixed input contraction argument in [Advanced COVE](ADVANCED.md#interacting-oscillatory-field) still applies in the maximum norm. The effective physical radius of one graph step changes with grid spacing. The experiments here establish numerical behavior and do not establish improved biological accuracy. Real and imaginary waveforms come from the already rotating complex state, while magnitude and wrapped phase are derived from that same state. The display does not apply an additional carrier rotation.

The feature atlas exposes sampled orientation magnitude, contrast and luminance along with a signed inverse Fourier projection of a selectable spatial band. All four projections sum to the processed luminance image with its mean removed. Point maps are plotted at the actual computed coordinates. The site by time and mode by time heatmaps may average adjacent temporal samples for display. The waveform rack retains every integration sample in its visible window.

## Local performance measurement

Median of three warm TypeScript runs in Node 24.19.0 on macOS arm64. Input was 18 decoded 64 × 64 WebM frames sampled at 12 frames per second for 162 integration steps. Times cover the encoder only and exclude video decoding, plotting and upload. Memory is the packed complex history only.

| Grid | Sites | Median ms | History MiB |
| --- | ---: | ---: | ---: |
| 5 × 5 | 25 | 24 | 0.25 |
| 10 × 10 | 100 | 40 | 0.99 |
| 16 × 16 | 256 | 77 | 2.53 |
| 32 × 32 | 1024 | 246 | 10.13 |

The 16 × 16 grid was selected as the default compromise for this workbench. The 32 × 32 grid was measured but is not a public selector option. Automated checks cover 10 × 10 and 16 × 16 local graph normalization and bounds, complex state dimensions, Python and TypeScript agreement at selected sites on every integration step, and Fourier projection consistency. The original THINGS EEG2 advanced report reproduced byte for byte.
