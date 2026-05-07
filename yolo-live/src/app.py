from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
import yaml
import os
import cv2
from pathlib import Path

# Local imports
from src.api.routes import router as api_router
from src.detector.model_loader import ModelLoader
from streams.camera import camera_generator

# --- PATH MANAGEMENT ---
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent

DEFAULT_CONFIG = ROOT_DIR / "config" / "config.yaml"
CONFIG_PATH = os.environ.get("CONFIG_PATH", str(DEFAULT_CONFIG))


def load_config(path=CONFIG_PATH):
    p = Path(path)

    if not p.exists():
        print(f"Warning: config not found at {p}. Trying fallback path.")
        alt_p = BASE_DIR.parent / "config" / "config.yaml"
        if alt_p.exists():
            p = alt_p
        else:
            return {"camera_source": 0, "backend": "opencv", "confidence": 0.25}

    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {"camera_source": 0, "backend": "opencv", "confidence": 0.25}


def create_app():
    app = FastAPI(title="yolo-live")

    cfg = load_config()

    # Avoid blocking app startup
    try:
        model_loader = ModelLoader(cfg)
    except Exception as e:
        print(f"Critical error in ModelLoader: {e}")
        model_loader = None

    app.include_router(api_router, prefix="/api")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/")
    async def viewer():
        template_path = BASE_DIR / "templates" / "viewer.html"
        if not template_path.exists():
            return JSONResponse(
                {
                    "error": "Template missing",
                    "requested_path": str(template_path),
                },
                status_code=404,
            )
        return FileResponse(template_path)

    def gen_frames():
        cam_src = cfg.get("camera_source", 0)

        for frame in camera_generator(cam_src):
            try:
                if frame is None or frame.size == 0:
                    continue

                # Optional AI inference:
                # if model_loader is not None:
                #     detections = model_loader.predict(frame)
                #     frame = model_loader.draw(frame, detections)

                ret, buffer = cv2.imencode(".jpg", frame)
                if not ret:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )

            except Exception as e:
                print(f"Error in gen_frames: {e}")
                continue

    from fastapi import BackgroundTasks

    @app.get("/stream")
    async def stream():
        return StreamingResponse(
            gen_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "Connection": "keep-alive"
            }
        )

    return app