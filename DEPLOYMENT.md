# Workbench deployment

Keep `Loreforall/bci-oscillatory-encoder` private. The Next.js build contains the public interface and API address. The Python service image and any mounted checkpoint should be accessible only to deployment operators.

## Why two services

[Vercel Functions limits](https://vercel.com/docs/functions/limitations) cap request bodies at 4.5 MB. The workbench accepts up to 20 MiB including multipart overhead. Full waveform JSON can also exceed 4.5 MB. Browser uploads and results therefore go directly to FastAPI over HTTPS. Vercel serves the static Next.js application.

The [Vercel Python runtime](https://vercel.com/docs/functions/runtimes/python) documents a 500 MB standard uncompressed bundle limit and a larger public-beta option. CPU-only Torch and headless OpenCV reduce the dependency footprint. A Linux function bundle, cold start, and decode latency would still need measurement on the target Vercel plan. A larger bundle allowance alone does not raise the request-body limit. Local timing and memory measurements do not establish serverless feasibility.

The installed Torch, OpenCV, and NumPy directories occupied about 781 MiB on the local Apple Silicon environment. That is a local package footprint. The Linux CPU-only artifact may differ. A local synthetic request at 16 channels and 500 Hz took 0.068 s with 266 MiB peak process RSS, including the test client and imported libraries. A low-complexity 120-frame 1080p MJPEG request at the same settings took 0.356 s and returned 10.6 MB of JSON. These single-run measurements exclude network transfer and platform cold starts. Codec, scene complexity, CPU and cold-start performance require verification on the selected host.

## Local development

From the repository root, using Python 3.12 and Node 24:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
uvicorn backend.api:app --port 8000
```

In a second terminal:

```sh
cd frontend
cp .env.example .env.local
npm ci
npm run dev
```

Open [the local workbench](http://localhost:3000). The existing `python app.py` Gradio entry point remains available.

## 1. Create the Vercel project

In the Vercel dashboard, import the private GitHub repository. Set **Root Directory** to `frontend`, **Framework Preset** to Next.js, and **Node.js Version** to 24.x. Leave the install and build commands at their defaults (`npm ci` with the committed lockfile and `npm run build`). Use `out` as the output directory if the project settings require an explicit directory. Choose a stable production domain to use in the backend's allowed origins. For a public release, configure Deployment Protection so the production domain opens without a Vercel login. The first functional production build needs the API URL from step 2.

## 2. Deploy the Python service

On a container host that can build from the private repository, select the root `Dockerfile` and route HTTPS traffic to port 8000. Configure the health check as `GET /health`. Start with one worker and a 2 GB CPU instance, then measure on the chosen host with representative media. The image installs the Linux CPU Torch build and runs as a non-root user. The build context excludes checkpoints, Git history, and local environment files.

| Backend variable | Value |
| --- | --- |
| `WORKBENCH_ORIGINS` | Comma-separated exact frontend origins, without spaces or trailing slashes. Include the production domain and any explicitly permitted preview domains. Defaults to `http://localhost:3000`. |
| `ENGINE_REVISION` | Private repository commit SHA or release identifier for exported provenance. Defaults to `workbench-v1`. |
| `ENCODER_CHECKPOINT` | Optional path to a trusted, read-only checkpoint mounted privately at runtime. Leave unset for the baseline-only release. |

For a host that takes an image from a private registry, build from the repository root:

```sh
docker build --platform linux/amd64 -t REGISTRY/PROJECT/bci-encoder:RELEASE .
docker push REGISTRY/PROJECT/bci-encoder:RELEASE
```

Use a registry path and release tag you control. Configure the host to preserve a valid Content-Length, enforce the 20 MiB request cap at ingress, and apply per-client rate limits and request deadlines. The service limits decoded frames and per-frame pixels, and admits one encoding job at a time per process. CORS is a browser policy, so direct HTTP callers also need ingress limits. Keep the decoder dependencies patched and set container memory and CPU limits. Use a host with adequate request and response limits for media and full waveform JSON. Temporary video files are deleted after processing, and multipart uploads are closed after each request.

Optionally train with `python train.py --epochs 80 --output artifacts/model.pt` and evaluate with `python compare.py --checkpoint artifacts/model.pt`. The API re-evaluates the checkpoint at startup and enables it only when all three held-out teacher-reproduction metrics improve on the existing constant predictor. It uses eight channels. This gate measures computational teacher reproduction.

## 3. Configure and verify production

Set these Vercel variables for **Production** and deploy or redeploy. Public variables are compiled into the build, so changing them requires rebuilding.

| Vercel variable | Value |
| --- | --- |
| `NEXT_PUBLIC_ENCODER_URL` | Public HTTPS base URL of the Python service with no path suffix. |
| `NEXT_PUBLIC_RESEARCH_EMAIL` | Optional public collaboration address. |

Verify the API first:

```sh
curl -fsS https://API_HOST/health
curl -fsS -X POST -d '' 'https://API_HOST/encode?kind=synthetic&channels=2&sample_rate=100' -o synthetic-run.json
```

Open the production domain in a signed-out browser. Confirm **Engine online**, run the synthetic example, inspect plots, download both exports, and verify a short MP4 and a sorted image sequence. Verify the GRU is disabled for a baseline-only release. For a checkpoint release, also run the GRU and inspect the deterministic overlay and comparison metrics.

## Data and scope

Video uses the decoder-reported nominal frame rate. Variable-rate media is interpreted on a uniform frame grid. Image sequences use filename order and the supplied frame rate. Input media can contain private information. The API has no persistent input store, and the frontend holds results in memory. Keep host logging and ingress retention consistent with that behavior. The JSON export includes processed-frame hashes, engine versions, parameters and full numerical arrays. Template CSV contains frame, channel and band rows.

The workbench produces computational simulations in arbitrary units. Visual-to-oscillatory representations have not been validated as a means of delivering visual perception to the human brain.
