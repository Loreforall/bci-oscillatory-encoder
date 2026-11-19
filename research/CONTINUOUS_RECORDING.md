# Continuous cortical recording extension

The earlier cortical display plotted a single excitatory deviation per site and resonance over 208 ms. The inputs were held, states began at zero, and the damped oscillator converged toward a driven equilibrium. The display already retained the integration samples. The transient shape therefore originated in the model and in the single-mode observation. The short window made the effect more prominent.

## Model and observation

The extension keeps the existing visual front end and introduces a separate stochastic recording pathway. Each site contains four damped E–I resonators. In normalized model units a local pair follows

`dE/dt = −α E − ω I + F`, `dI/dt = ω E − α I`.

Substep spatial mixing uses a convex combination of neighboring states, followed by the exact locally forced rotational update over one 480 Hz interval. The local orientation quadratures at two spatial scales, contrast and frame motion determine bounded visual drive `d`. The drive affects force, damping, frequency and ongoing input gain. Rows of the spatial graph vary with local contrast. Before visual latency elapses the graph is stimulus independent. A warm-up runs for one second before the visible half-second baseline.

The force combines adapting visual drive with local and shared Gaussian Ornstein–Uhlenbeck processes. Each has update `ξₖ₊₁ = ρ ξₖ + √(1−ρ²) εₖ`, with unit Gaussian innovations and `ρ = exp(−Δt/τ)`. The four correlation times are 45, 27, 14 and 7 ms. The slow adaptation time constant is 240 ms. Gain also depends on the aggregate local excitatory state, coupling the resonators. The browser uses seed 1731. Stimulus latency is approximately 18 ms. For video, each decoded frame changes the corresponding visual drive during its display interval before the model delay.

The observation is a weighted sum across the four E–I pairs at each site, using an excitatory-minus-inhibitory current proxy. Single-pole 0.7 Hz high-pass and 85 Hz low-pass stages precede instantaneous common-average reference across the full field. No additional display noise is injected. This proxy is in arbitrary units and has no anatomical or volume-conduction model. A plotted site is a retinotopic location. It cannot be mapped to a scalp electrode. The cutoff frequencies are filter parameters, rather than brick-wall band limits. The nominal resonances are 8, 14, 24 and 38 Hz, increasing up to 16% as visual drive increases. The sampling Nyquist limit is 240 Hz. [Synaptic-current proxy background](https://doi.org/10.1371/journal.pcbi.1004584)

The waveform, absolute instantaneous spatial map, windowed power, Welch power density, coherence and correlation all read the same filtered common-referenced samples. The waveform uses every sample. Display normalization is per row. Temporal diagnostics use Hann windows and cover the baseline plus up to three seconds of post-onset data. The original COVE states and prior ERP pilot remain separate.

## Checks

The browser's exact neighborhood raster was compared against gray and the THINGS aircraft-carrier photo at the same seed. The visible baseline matched bit for bit. Post-onset RMS differences were 0.149 and 0.085 a.u. respectively. Changing the seed for the same photograph gave 0.402 a.u. RMS difference. The final half second retained 0.296 a.u. RMS activity.

A descriptive comparison used the first 750 ms after onset at eight spatially distributed simulated sites and eight posterior channels from participant 01's verified THINGS-EEG2 trials. Both spectra used half-second Hann windows. The simulated trace had 14% of its 2–86 Hz power below 8 Hz, compared with 36% in recorded EEG. Median absolute interchannel correlation was 0.114 and 0.369, respectively. Signal variance is in different units and is included for inspection only. The signals use different references and no matched neighborhood recording is available. This check does not validate the simulation against human physiology. The existing EEG prediction benchmark is unaffected. [THINGS-EEG2 dataset](https://doi.org/10.1016/j.neuroimage.2022.119754)

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/cove-mpl \
  .venv/bin/python -m research.recording_check \
  --output /tmp/cove-recording.json --figure /tmp/cove-recording-check.png
```

The optional figure shows one channel overlaid across the matched-seed image cases and 16 channels from the neighborhood trace. [Recorded metrics](results/recording.json)
