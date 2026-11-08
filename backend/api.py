"""Public, bounded interface to the private research engine."""
import hashlib
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", "2073600")

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from baseline import BAND_FREQUENCIES, FEATURE_NAMES, encode_features
from model import SensoryOscillationEncoder
from synthesizer import WaveformSynthesizer
from train import loss_parts, make_batch
from visual_features import VisualFeatureExtractor

MAX_BYTES = 20 * 1024**2
MAX_FRAMES = 120
MAX_PIXELS = 1920 * 1080
FEATURE_SIZE = (64, 64)
logger = logging.getLogger(__name__)


class InputError(ValueError):
    """A media error whose message can be returned to clients."""


@torch.inference_mode()
def load_checkpoint(path):
    model = SensoryOscillationEncoder().eval()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    x, targets = make_batch(np.random.default_rng(20261108), batch=128)
    _, metrics = loss_parts(model(x), targets)
    constant = {k: v.mean(dim=(0, 1), keepdim=True).expand_as(v) for k, v in targets.items()}
    _, reference = loss_parts(constant, targets)
    if not all(np.isfinite(value) and value < reference[key] for key, value in metrics.items()):
        raise ValueError("Checkpoint failed the held-out synthetic benchmark")
    return model, {"sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                   "held_out": metrics, "constant_reference": reference}


@asynccontextmanager
async def lifespan(app):
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    app.state.model = None
    app.state.checkpoint = None
    app.state.busy = False
    path = os.getenv("ENCODER_CHECKPOINT")
    if path:
        try:
            app.state.model, app.state.checkpoint = load_checkpoint(path)
        except Exception:
            logger.exception("Configured checkpoint could not be enabled")
    yield


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)



@app.middleware("http")
async def bound_run(request, call_next):
    if request.method != "POST" or request.url.path != "/encode":
        return await call_next(request)
    length = request.headers.get("content-length", "")
    if request.headers.get("transfer-encoding") or not length.isascii() or not length.isdecimal() or len(length) > 9:
        return JSONResponse({"detail": "A valid Content-Length is required."}, 411)
    if int(length) > MAX_BYTES:
        return JSONResponse({"detail": "Upload exceeds the 20 MiB request limit."}, 413)
    if app.state.busy:
        return JSONResponse({"detail": "Encoder busy. Retry shortly."}, 429)
    app.state.busy = True
    try:
        return await call_next(request)
    finally:
        app.state.busy = False


@app.get("/health")
def health():
    return {"status": "ok", "learned": app.state.model is not None,
            "max_frames": MAX_FRAMES, "max_bytes": MAX_BYTES, "bands_hz": BAND_FREQUENCIES}


def dimensions(width, height):
    if width <= 0 or height <= 0 or width * height > MAX_PIXELS:
        raise InputError("Input resolution must be at most 2,073,600 pixels per frame.")
    return width, height


def read_video(upload):
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "input")
        with open(path, "wb") as output:
            while chunk := upload.file.read(1024 * 1024):
                output.write(chunk)
        capture = cv2.VideoCapture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
        try:
            if not capture.isOpened():
                raise InputError("Video could not be opened. Try an MP4 or WebM file.")
            size = dimensions(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not 0.01 <= fps <= 1000:
                raise InputError("Video must report a frame rate between 0.01 and 1000 fps.")
            frames = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if len(frames) == MAX_FRAMES:
                    raise InputError("Video exceeds 120 decoded frames. Trim it before uploading.")
                if dimensions(frame.shape[1], frame.shape[0]) != size:
                    raise InputError("Video resolution must remain constant.")
                frames.append(cv2.cvtColor(cv2.resize(frame, FEATURE_SIZE), cv2.COLOR_BGR2RGB))
            return frames, size, fps
        finally:
            capture.release()


def read_images(uploads):
    frames = []
    size = None
    for upload in sorted(uploads, key=lambda item: item.filename):
        with Image.open(upload.file, formats=["PNG", "JPEG", "WEBP"]) as image:
            current_size = dimensions(*image.size)
            if size is not None and current_size != size:
                raise InputError("All images must have the same resolution.")
            if getattr(image, "is_animated", False):
                raise InputError("Upload still images for an image sequence.")
            size = current_size
            frames.append(cv2.resize(np.asarray(image.convert("RGB")), FEATURE_SIZE))
    return frames, size


def synthetic():
    y, x = np.mgrid[:96, :128]
    frames = []
    for index in range(48):
        luminance = 128 + 65 * np.sin(x / 4 - index / 3) * np.cos(y / 8)
        luminance += 25 * np.sin(index / 7)
        frames.append(np.repeat(luminance.astype(np.uint8)[..., None], 3, axis=2))
    return frames, (128, 96), 24.0


@torch.inference_mode()
def run(kind, uploads, encoder, channels, fps, sample_rate):
    if kind == "synthetic":
        frames, size, fps = synthetic()
    elif kind == "video":
        frames, size, fps = read_video(uploads[0])
    else:
        frames, size = read_images(uploads)
    if not frames:
        raise InputError("No usable frames were decoded.")

    features = VisualFeatureExtractor(FEATURE_SIZE).extract_sequence_features(frames)
    baseline = {key: torch.from_numpy(value).unsqueeze(0)
                for key, value in encode_features(features, channels).items()}
    template = baseline
    comparison = None
    if encoder == "learned":
        template = app.state.model(torch.as_tensor(features, dtype=torch.float32).unsqueeze(0))
        _, comparison = loss_parts(template, baseline)
    synthesizer = WaveformSynthesizer(sampling_rate=sample_rate, duration=0.5)
    wave = synthesizer.synthesize(template)[0]
    motion = features[:, 7]
    spectrum = abs(np.fft.rfft(motion - motion.mean()))**2 / len(motion)**2

    return {
        "schema_version": 1,
        "signal_type": "simulated oscillations",
        "input": {"kind": kind, "width": size[0], "height": size[1], "frames": len(frames), "fps": fps,
                  "processed_frames_sha256": hashlib.sha256(np.stack(frames).tobytes()).hexdigest()},
        "parameters": {"encoder": encoder, "channels": channels, "bands_hz": BAND_FREQUENCIES,
                       "sample_rate_hz": sample_rate, "window_seconds": 0.5, "feature_size": FEATURE_SIZE},
        "runtime": {"numpy": np.__version__, "opencv": cv2.__version__, "torch": str(torch.__version__),
                    "engine_revision": os.getenv("ENGINE_REVISION", "workbench-v1")},
        "checkpoint": app.state.checkpoint if encoder == "learned" else None,
        "feature_names": FEATURE_NAMES + tuple(f"temporal_{name}" for name in FEATURE_NAMES),
        "features": features.tolist(),
        "motion_spectrum": {"hz": np.fft.rfftfreq(len(motion), 1 / fps).tolist(), "power": spectrum.tolist()},
        "template": {key: value[0].tolist() for key, value in template.items()},
        "baseline": {key: value[0].tolist() for key, value in baseline.items()} if comparison else None,
        "waveforms": wave.tolist(),
        "metrics": {"waveform_rms": float(wave.square().mean().sqrt()), "peak_absolute": float(wave.abs().max()),
                    "mean_band_power": template["band_power"].mean(dim=(0, 1, 2)).tolist()},
        "comparison": comparison,
    }


@app.post("/encode")
async def encode(request: Request, kind: str = Query(pattern="^(synthetic|video|images)$"),
                 encoder: str = Query("baseline", pattern="^(baseline|learned)$"),
                 channels: int = Query(8, ge=1, le=16), fps: float = Query(24, ge=1, le=120),
                 sample_rate: int = Query(250, ge=100, le=500)):
    if encoder == "learned" and (app.state.model is None or channels != 8):
        raise HTTPException(422, "Learned encoding requires an enabled checkpoint and eight channels.")
    try:
        if kind == "synthetic":
            result = await run_in_threadpool(run, kind, [], encoder, channels, fps, sample_rate)
        else:
            if not request.headers.get("content-type", "").startswith("multipart/form-data;"):
                raise HTTPException(415, "Upload media as multipart form data.")
            async with request.form(max_files=MAX_FRAMES if kind == "images" else 1, max_fields=0) as form:
                uploads = form.getlist("files")
                if (not uploads or len(uploads) != len(form.multi_items())
                        or any(not isinstance(upload, UploadFile) for upload in uploads)):
                    raise HTTPException(422, "Attach media using the files field.")
                result = await run_in_threadpool(run, kind, uploads, encoder, channels, fps, sample_rate)
        return await run_in_threadpool(JSONResponse, result, headers={"Cache-Control": "no-store"})
    except UnidentifiedImageError:
        raise HTTPException(422, "Use supported PNG, JPEG or WebP still images.") from None
    except InputError as error:
        raise HTTPException(422, str(error)) from None
    except (cv2.error, OSError, Image.DecompressionBombError):
        raise HTTPException(422, "Media decoding failed or the input exceeds decoder limits.") from None


app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("WORKBENCH_ORIGINS", "http://localhost:3000").split(","),
    allow_methods=["GET", "POST"],
)
