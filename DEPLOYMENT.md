# Vercel deployment

Vercel serves a static Next.js build. The browser reads selected media, extracts visual features, runs the deterministic encoder, synthesizes waveforms, and generates exports locally. The public workbench needs no API URL, server function, checkpoint, database, or external compute host.

## Deploy from the private repository

1. Import `Loreforall/bci-oscillatory-encoder` into Vercel and select `main` as the production branch. Keep the GitHub repository private.
2. Set **Root Directory** to `frontend`, **Framework Preset** to **Next.js**, and **Node.js Version** to **24.x**. The default build command is `npm run build`. Next.js static export writes `out/`.
3. Optionally set `NEXT_PUBLIC_RESEARCH_EMAIL` to the public address for collaboration inquiries. Rebuild after changing it.
4. Deploy. Open the production domain signed out, run the moving grating, select a channel and band, and export CSV and JSON. For media, test a short browser-compatible MP4 or WebM and a sequence of PNGs.

For a CLI deployment to a project already configured with this root directory, stage only the committed frontend in a temporary directory and deploy from its parent:

```sh
release_dir=$(mktemp -d)
git archive HEAD frontend | tar -x -C "$release_dir"
vercel deploy "$release_dir" --project bci-oscillatory-encoder --prod --yes
```

## Runtime assessment

The existing deterministic encoder in `baseline.py` depends only on NumPy and could run by itself in a Vercel Python Function. The complete Python service also imports OpenCV and Torch. Vercel's [Python runtime](https://vercel.com/docs/functions/runtimes/python) has a 500 MB standard uncompressed bundle limit and a larger public-beta option. The earlier local macOS install occupied about 781 MiB across NumPy, OpenCV and Torch. The Linux CPU bundle was not measured. Moving the small mapping alone to a function would still leave browser feature extraction and synthesis to implement and would require transmitting features off-device.

[Vercel Functions](https://vercel.com/docs/functions/limitations) cap request and ordinary response bodies at 4.5 MB. The former API accepted 20 MiB uploads and a local 120-frame test returned 10.6 MB of JSON. Browser computation avoids those payload limits and keeps selected media and results on-device. The frontend does not include the original Python modules or GRU weights. The TypeScript port is delivered as browser JavaScript.

## Scientific scope and input limits

- The public app computes the four-band deterministic template. GRU inference and baseline-versus-GRU comparison remain available in local Python research.
- The numerical port uses the same feature definitions, Canny thresholds, FFT band cutoff, temporal filters, mapping and synthesis formulas. Floating-point precision, Canny threshold boundaries, codec behavior, color conversion and browser resizing can cause differences from the Python pipeline. Numerical parity tests use identical, already-resized RGB pixels.
- Image sequences use lexicographic filename order. Select same-size PNG, JPEG or WebP stills. Transparent pixels are composited on black. Browser decoding can apply orientation and color profiles. Animated images are outside the supported input contract.
- Video uses the browser's available codecs and samples the displayed frame at uniform seek times starting at zero. The configured rate describes sample times. Depending on native frame rate and timestamps, samples can repeat or skip source frames. The source's native frame count and FPS are not used.
- The app bounds files to 20 MiB total, 120 analysis frames, and 2,073,600 pixels per decoded frame. At 12 samples/s, use clips up to about 10 seconds. Lower the sampling rate or trim longer clips. Runtime depends on the device and codec.
- Use an HTTPS or localhost origin for the processed RGB SHA-256 hash. JSON also includes the analysis timebase, browser, engine version and parameters. Exports describe simulated oscillations in arbitrary units.

## Local checks

```sh
# Python 3.12, from the repository root after installing requirements.txt
python -m pytest -q
python -m tests.generate_browser_reference

# Node 24
cd frontend
npm ci
npm test
npm run build
npm run check
npm run dev
```

The reference generator evaluates the original Python extractor, NumPy encoder and Torch synthesizer. The TypeScript tests check all 16 features and template values, motion spectra, selected waveform samples, RMS and peak. The fixture includes constant color, straight and diagonal edges, RGB texture and checkerboards, plus clipping and one-channel cases. Tests use absolute tolerances of 2e-6 for features, 1e-6 for template values, and 5e-6 for waveform samples and metrics.
