# Advanced COVE encoder

Advanced COVE keeps two synchronized representations. A 25-site complex field supports oscillatory analysis and inverse reconstruction. A visual bypass retains site features, quadrature coefficients, spatial band energies and downsampled pixels for calibration. The original workbench retains its encoder, plots, and exports. **Inspect distributed COVE**, under the plots after a run, opens the advanced panel.

## Reproduce the implementation and evidence

Use the [pinned Python environment and public data acquisition](README.md#reproduce). The original report describes which THINGS-EEG2 derivatives and stimulus tables were aligned. From the repository root:

```bash
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python -m research.advanced_run
python -m research.decoder
python -m tests.generate_advanced_reference
python -m research.calibrate --subject 01 --out research-data/sub-01-v2.json
python -m research.calibrate --subject 01 --hybrid --out research-data/sub-01-hybrid.json
python -m pytest -q
cd frontend
npm test
npm run check
npm run build
```

Run a still image, open **Inspect distributed COVE**, and load either calibration under **ERP calibration JSON**. The default export chooses among visual and hybrid models on validation. The `--hybrid` export requires an oscillatory block. `--legacy` exports the original v1 model, and the workbench still reads v1 files. The decoder works on images and videos without a calibration. ERP calibration is scoped to still-image onset, the available experimental condition. Participant calibrations remain local. The demo's original RGB preview preserves color at its processing resolution. Advanced filtering, calibration and reconstruction use luminance. Browser image decoding and resize can differ from OpenCV; parity checks begin with identical RGB pixels.

## Visual operators

Let $G_t\in[0,1]^{64\times64}$ be a luminance frame. The Fourier block uses the 2D DFT of $G_t-\overline G_t$. For radial frequencies in four bins with boundaries $1/16$, $1/8$ and $1/4$ cycles per pixel, the feature is

$$s_j=\sqrt{\sum_{(u,v)\in B_j}|\widehat G_t(u,v)|^2}/64^2.$$

The squared features sum to frame variance by Parseval. The browser shares the original encoder's FFT and retains its original low/high energy output. Fourier magnitude discards global phase, so the bypass also retains signed local quadrature and spatial pixels.

The existing exponential radial grid, local contrast normalization and eccentricity-dependent high-frequency attenuation provide central and peripheral sampling. Each site has low and high scale sine and cosine responses at four orientations. The 375-dimensional conventional bypass keeps the original luminance, contrast, luminance motion, quadrature phase-change proxy, and Gabor amplitudes. The 548-dimensional detail bypass adds 400 signed quadrature values, four spatial band RMS features multiplied by 4, and 144 area-averaged grayscale pixels. Browser and Python use the same coefficient order. The Gabor shapes and base drive are specified in the [original mathematical notes](README.md#visual-representation).

For site luminance $L_t$, the causal undecimated Haar detail at scale $m\in\{1,2,4\}$ is

$$H_m(t)=\frac{1}{m\sqrt 2}\sum_{r=0}^{m-1}(L_{t-r}-L_{t-m-r}).$$

Negative indices use the first frame; stills give zero detail. Each coefficient measures contrast between adjacent time windows of length $m/f_s$. Its response spans a frequency band and depends on the frame rate. Add $0.25i[H_4,H_2,H_1,H_1-H_2]$ to the four original complex drives and apply $u/\max(1,|u|)$. The original signed frame difference and quadrature phase change remain active. Reconstruction and EEG fit preserve additional spatial detail outside this bounded projection.

## Interacting oscillatory field

Let $W(c)$ be the existing local row-stochastic graph operator, with logits $-a\|p_i-p_j\|_1-b|c_i-c_j|$. In the demo the default slopes are $(2,2)$. The calibration export uses the training-only masked-prediction slopes, which were $(0,2)$ in the pilot. The row weights respond to image contrast. Let $d$ be the bounded visual drive, $D=.8d+.2Wd$, $h=1/(f_s\lceil100/f_s\rceil)$, and $\rho=e^{-h/.09}$. With mode index $k\in\{0,1,2,3\}$ and $\omega_k=2\pi(6,10,20,40)_k$, define

$$
\theta_{ik}=h[\omega_k+2.4k\,\operatorname{Im}(z_{i0})],\qquad
g_{ik}=1-.09\,\mathbf 1_{k>0}[1-\operatorname{Re}(z_{i0})],
$$

$$z'_{ik}=\rho e^{i\theta_{ik}}(Wz_k)_i+(1-\rho)g_{ik}D_{ik}.$$

The high-mode angular rates depend on the low mode's quadrature. Its in-phase coordinate $|z_{i0}|\cos(\arg z_{i0})$ also gates higher-mode drive amplitudes. This specifies phase-amplitude interaction without treating phase as reliable when low-mode amplitude vanishes. Both interactions act within a frame. The displayed coupling gain and instantaneous frequency shift are evaluated at the selected site's current state.

The useful field-theoretic analogy is a classical complex mode split. For frozen low-mode coordinates, a quadratic generator $\mathcal H=\sum_{ik}(\omega_k+2.4k\,\operatorname{Im}z_{i0})|Z_{ik}|^2$ produces a norm-preserving rotation for independent $Z_{ik}$ under $\dot Z=i\partial\mathcal H/\partial\bar Z$. The implemented one-way modulation freezes the low coordinate for each substep and uses its exact phase rotation. The full driven update is a classical dissipative system. For fixed contrast, the symmetric unnormalized graph weights $A$ and degrees $q_i=\sum_j A_{ij}$ give $W=\operatorname{diag}(q)^{-1}A$. Jensen's inequality gives $\sum_i q_i |(Wz)_i|^2\le\sum_i q_i|z_i|^2$, so the graph mixing dissipates spatial differences in the degree-weighted energy. With time-varying contrast, the unweighted maximum-modulus bound gives the uniform guarantee.

Since $0.82\le g\le1$, the disk per complex coordinate is invariant. In max modulus, the same-input map has Lipschitz bound

$$Q(h)=\rho(1+7.2h)+.09(1-\rho)<1.$$

The phase term uses $|e^{ia}-e^{ib}|\le |a-b|$ with $k\le3$; the drive gate has Lipschitz constant .09. The strict bound follows from $1-\rho\ge\rho h/.09$ and $7.2<.91/.09$. The bound applies to initial states in the polydisc and fixed input sequences. Frame outputs retain all four complex modes at each site. The browser also samples summed real site states at the internal integration rate and plots Hann-windowed spectral power and a spectrogram. The spectra describe modeled trajectories in arbitrary units.

## Onset-aware EEG readout

The calibration target is eight posterior-channel ERPs with 30 post-trigger bin centers from 10 to 590 ms. Each condition uses 80 averaged repeats of the human recording. The still-image encoder starts at zero and holds the drive for five 24 fps intervals before constructing the fixed-dimensional state code. Output coefficients are indexed by sensor and post-onset time. Source-space interpretation, single-trial prediction, and valid predictions for continuous video would require separate evidence.

For visual and state blocks $x_b$ with training means $\mu_b$ and standard deviations $\sigma_b$ floored at .01, use Gaussian similarities

$$K_b(x,x')=\exp[-\|(x-x')/\sigma_b\|^2/(d_b\ell)],\qquad K=\sum_b w_bK_b.$$

The six nonnegative mixtures in [hybrid.py](hybrid.py) include visual-only controls, light hybrids, and larger state weights. Each $K_b$ is positive semidefinite and mixtures preserve that property. Center the Gram matrix on training conditions and solve $(K_c+\lambda I)A=Y-\bar Y$ in its eigensystem. Validation selects $\ell\in\{.5,2,8\}$, $\lambda\in\{.01,.1,1,10\}$, mixture, and optional temporal regularization. For post-onset predictions, $(I+\eta D_2^\top D_2)^{-1}$ penalizes second differences of the fitted residual trace per channel, preserving affine time trends; $\eta\in\{0,.4,3\}$. Validation selected zero smoothing for the reported ERP fits. The precomputed smoothing and Gram centering are folded into the exported dual weights and intercept. No test targets enter coefficient fitting, feature normalization, or selection.

The comparison repeats the previous 120/40/40 concept splits at seeds 17, 29 and 43. The original 200-feature nonlinear visual baseline was reproduced on the same splits. The scores below are average percent MSE skill against the training mean.

| Readout | Sub-01 ERP | Sub-02 ERP | Sub-01 log power | Sub-02 log power |
| --- | ---: | ---: | ---: | ---: |
| Previous full COVE | 0.32% | 0.40% | −0.96% | −2.76% |
| Previous nonlinear visual baseline | 8.09% | 10.47% | −1.07% | −0.27% |
| Conventional visual kernel | 10.76% | 13.01% | 0.02% | −0.01% |
| Validation-selected hybrid | 10.75% | 12.95% | 0.17% | −0.93% |
| Validation-selected calibration | 10.58% | 13.01% | 0.17% | −0.93% |

The ERP improvement over the previous nonlinear baseline persisted with the light state mixture. The simpler visual kernel achieved comparable or slightly higher mean skill. An independent benefit from oscillatory features remains unresolved. The default calibrator selected the conventional visual kernel in five of six ERP fits and a detailed visual kernel in the other. Log-power gains were inconsistent, and the demonstration exports ERP only. [Full results](results/hybrid.json) include time-resolved errors and paired within-split bootstrap intervals; the three splits overlap and include two people.

## Inverse and measured stability

The portable grayscale renderer $R$ is an affine ridge model from 200 real state coordinates to a 12 × 12 image, trained on 100 procedural eight-frame clips (seed 401). For an input clip the forward decode is $Z_0R+r$. The browser then solves a constrained refinement against known input pixels $P$ in contiguous windows of up to eight frames. Put $U=Z-Z_0$ and use

$$
\mathcal L=\|ZR+r-P\|_F^2/144
+.06\|U\|_F^2+.02\|\Delta_t U\|_F^2+.035\|(I-W)U\|_F^2 .
$$

The graph term is evaluated separately per frame. Each of 28 iterations uses the gradient and a conservative reciprocal Hessian bound $[2\|R\|_2^2/144+.28+.07(1+\sqrt5)^2]^{-1}$. This uses $\|\Delta_t\|_2^2\le4$ and $\|W\|_2\le\sqrt5$ from row sums of one and at most five neighbors per column. Each complex displacement is capped at .15, then shortened along its anchor ray at the unit-circle intersection when required. This bounds state changes and numerical amplitude. It is a feasible radial update with no global optimality claim. The renderer reports raw RMSE and peak-1 PSNR against downsampled input, and clamps only the displayed grayscale. This optimization uses visual pixels. The ERP prediction uses the original forward features, preserving the benchmarked calibration path.

| Evaluation input | Forward RMSE | Refined RMSE | Refined PSNR |
| --- | ---: | ---: | ---: |
| 24 unseen procedural clips, seed 402 | 0.1621 | 0.0679 | 23.36 dB |
| 200 THINGS-EEG2 test-pool images | 0.2301 | 0.1203 | 18.39 dB |

Frame-difference RMSE on procedural clips fell from 0.0256 to 0.0121. The largest refined amplitude was 0.198 across these reconstruction runs. A same-input perturbation test decreased maximum complex-state separation from 1.4142 to 0.0349 in eight frames. [Detailed measurements](results/decoder.json) distinguish numerical behavior, visual reconstruction, and separately scored EEG prediction. Natural-image forward error remains substantial because the small renderer learns on procedural material. Refined images use an input-dependent adjustment and are not evidence of reconstructing an unknown image from recorded EEG.
