from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
import yaml
import os
import cv2
from pathlib import Path
import logging

# Try relative imports with fallback to top-level imports
try:
    from src.api.routes import router as api_router
except Exception:
    try:
        from api.routes import router as api_router
    except Exception:
        api_router = None

try:
    from src.detector.model_loader import ModelLoader
except Exception:
    try:
        from detector.model_loader import ModelLoader
    except Exception:
        ModelLoader = None

# camera_generator may live under src.streams or streams
camera_generator = None
for candidate in ("src.streams.camera", "streams.camera", "src.camera_streams.camera", "camera_streams.camera"):
    try:
        module = __import__(candidate, fromlist=["camera_generator"])
        camera_generator = getattr(module, "camera_generator")
        break
    except Exception:
        continue

# --- PATH MANAGEMENT ---
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent

DEFAULT_CONFIG = ROOT_DIR / "config" / "config.yaml"
CONFIG_PATH = os.environ.get("CONFIG_PATH", str(DEFAULT_CONFIG))

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yolo-live")


def load_config(path=CONFIG_PATH):
    p = Path(path)

    if not p.exists():
        logger.warning(f"Config not found at {p}. Trying fallback path.")
        alt_p = BASE_DIR.parent / "config" / "config.yaml"
        if alt_p.exists():
            p = alt_p
        else:
            logger.warning("No config found; using defaults.")
            return {"camera_source": 0, "backend": "opencv", "confidence": 0.25}

    try:
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            # Ensure defaults exist
            cfg.setdefault("camera_source", 0)
            cfg.setdefault("backend", "opencv")
            cfg.setdefault("confidence", 0.25)
            return cfg
    except Exception as e:
        logger.error(f"Failed to read config {p}: {e}")
        return {"camera_source": 0, "backend": "opencv", "confidence": 0.25}


def create_app():
    app = FastAPI(title="yolo-live")

    cfg = load_config()

    # Initialize model loader but don't fail startup if it errors
    model_loader = None
    if ModelLoader is not None:
        try:
            model_loader = ModelLoader(cfg)
            logger.info("ModelLoader initialized.")
        except Exception as e:
            logger.error(f"Critical error initializing ModelLoader: {e}")
            model_loader = None
    else:
        logger.info("ModelLoader class not available; skipping model initialization.")

    if api_router is not None:
        app.include_router(api_router, prefix="/api")
    else:
        logger.info("API router not available; skipping router include.")

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
        return FileResponse(template_path, media_type="text/html")

    def gen_frames():
        cam_src = cfg.get("camera_source", 0)

        # If camera_generator is not available, try to open cv2.VideoCapture as fallback
        use_cv_capture = camera_generator is None

        if use_cv_capture:
            logger.info("camera_generator not found; using cv2.VideoCapture fallback.")
            try:
                cap = cv2.VideoCapture(int(cam_src) if str(cam_src).isdigit() else cam_src)
            except Exception as e:
                logger.error(f"Failed to open VideoCapture with source {cam_src}: {e}")
                cap = None
        else:
            cap = None

        try:
            if use_cv_capture and (cap is None or not cap.isOpened()):
                logger.error("No camera available for streaming.")
                return

            # If using camera_generator, iterate it; otherwise read frames from cap
            if not use_cv_capture:
                for frame in camera_generator(cam_src):
                    try:
                        if frame is None:
                            continue
                        if hasattr(frame, "size") and frame.size == 0:
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
                        logger.exception(f"Error processing frame from camera_generator: {e}")
                        continue
            else:
                while True:
                    try:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            continue

                        # Optional AI inference:
                        # if model_loader is not None:
                        #     detections = model_loader.predict(frame)
                        #     frame = model_loader.draw(frame, detections)

                        ret2, buffer = cv2.imencode(".jpg", frame)
                        if not ret2:
                            continue

                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                        )
                    except Exception as e:
                        logger.exception(f"Error reading frame from cv2.VideoCapture: {e}")
                        continue
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    @app.get("/stream")
    async def stream():
        # StreamingResponse will stream multipart JPEG frames
        return StreamingResponse(
            gen_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "Connection": "keep-alive",
            },
        )

    return app
