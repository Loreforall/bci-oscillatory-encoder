from io import BytesIO

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend.api import MAX_BYTES, app


def test_synthetic_run_and_checkpoint_gate(monkeypatch):
    monkeypatch.delenv("ENCODER_CHECKPOINT", raising=False)
    with TestClient(app) as client:
        assert client.get("/health").json()["learned"] is False
        assert client.post("/encode?kind=synthetic&encoder=learned", content=b"").status_code == 422
        response = client.post("/encode?kind=synthetic&channels=2&sample_rate=100", content=b"")
        assert response.status_code == 200
        result = response.json()
        assert result["input"]["frames"] == 48
        assert np.asarray(result["features"]).shape == (48, 16)
        assert np.asarray(result["waveforms"]).shape == (48, 2, 50)
        assert result["metrics"]["waveform_rms"] > 0
        assert result["checkpoint"] is None
        assert client.get("/openapi.json").status_code == 404


def test_upload_limits_and_image_order():
    def png(brightness):
        output = BytesIO()
        Image.new("RGB", (64, 32), (brightness,) * 3).save(output, format="PNG")
        return output.getvalue()

    with TestClient(app) as client:
        too_large = client.post("/encode?kind=images", headers={"Content-Length": str(MAX_BYTES + 1)})
        assert too_large.status_code == 413
        invalid = client.post("/encode?kind=images", files={"files": ("x.png", b"invalid", "image/png")})
        assert invalid.status_code == 422
        images = [("files", ("02.png", png(255), "image/png")), ("files", ("01.png", png(0), "image/png"))]
        response = client.post("/encode?kind=images&fps=12", files=images)
        assert response.status_code == 200
        data = response.json()
        assert data["input"]["width"] == 64
        assert data["input"]["fps"] == 12
        assert [frame[0] for frame in data["features"]] == [0, 1]


def test_video_upload(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15, (64, 32))
    assert writer.isOpened()
    try:
        for brightness in (16, 96, 192):
            writer.write(np.full((32, 64, 3), brightness, dtype=np.uint8))
    finally:
        writer.release()
    with TestClient(app) as client:
        response = client.post("/encode?kind=video", files={"files": ("clip.avi", path.read_bytes(), "video/x-msvideo")})
        assert response.status_code == 200
        data = response.json()
        assert data["input"]["frames"] == 3
        assert data["input"]["fps"] == 15


def test_learned_accepts_extracted_features(monkeypatch):
    from backend import api
    from model import SensoryOscillationEncoder

    # Isolate inference integration from the separate trained-checkpoint benchmark.
    monkeypatch.setenv("ENCODER_CHECKPOINT", "test-fixture")
    monkeypatch.setattr(api, "load_checkpoint", lambda _: (SensoryOscillationEncoder().eval(), {}))
    with TestClient(app) as client:
        response = client.post("/encode?kind=synthetic&encoder=learned", content=b"")
        assert response.status_code == 200
        result = response.json()
        assert result["baseline"] is not None
        assert set(result["comparison"]) == {"power_mse", "phase_circular", "spatial_mse"}
