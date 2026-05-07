from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
import yaml, os, asyncio
from pathlib import Path
from src.api.routes import router as api_router
from src.detector.model_loader import ModelLoader
from src.streams.camera import camera_generator, camera_available

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = os.environ.get("CONFIG_PATH", BASE_DIR.parent / "config" / "config.yaml")

def load_config(path=CONFIG_PATH):
    if isinstance(path, Path):
        path = str(path)
    with open(path, "r") as f:
        return yaml.safe_load(f)

def create_app():
    app = FastAPI(title="yolo-live")

    cfg = load_config()
    model_loader = ModelLoader(cfg)

    app.include_router(api_router, prefix="/api")

    @app.get("/health")
    async def health():
        return JSONResponse({"status":"ok"})

    @app.get("/status")
    async def status():
        cam_src = cfg.get("camera_source", 0)
        return JSONResponse({
            "camera_source": cam_src,
            "camera_available": camera_available(cam_src),
            "backend": cfg.get("backend", "yolov8"),
        })

    @app.get("/")
    async def viewer():
        return FileResponse(BASE_DIR / "templates" / "viewer.html", media_type="text/html")

    def gen_frames():
        cam_src = cfg.get("camera_source", 0)
        for frame in camera_generator(cam_src):
            # run detection
            detections = model_loader.predict(frame)
            out = model_loader.draw(frame, detections)
            # encode frame as JPEG
            import cv2
            ret, buffer = cv2.imencode('.jpg', out)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    @app.get("/stream")
    def stream():
        return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    return app