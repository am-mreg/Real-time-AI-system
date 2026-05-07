from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
import yaml
import os
import asyncio
import cv2 
from pathlib import Path

# Importet lokale
# Duke qenë se jemi brenda 'src', importojmë direkt nga modulet fqinje
from src.api.routes import router as api_router
from src.detector.model_loader import ModelLoader
from src.streams.camera import camera_generator, camera_available

# --- MENAXHIMI I SHTIGJEVE (Sipas strukturës tënde) ---
# BASE_DIR = .../Real-Time-AI-System/yolo-live/src
BASE_DIR = Path(__file__).resolve().parent

# Shkojmë te Real-Time-AI-System (3 nivele lart nga app.py)
# app.py -> src -> yolo-live -> Real-Time-AI-System
ROOT_DIR = BASE_DIR.parent.parent 

# Nëse folderi 'config' është te Real-Time-AI-System/config/
DEFAULT_CONFIG = ROOT_DIR / "config" / "config.yaml"
CONFIG_PATH = os.environ.get("CONFIG_PATH", DEFAULT_CONFIG)

def load_config(path=CONFIG_PATH):
    p = Path(path)
    if not p.exists():
        print(f"Kujdes: Config nuk u gjet në {p}. Po përdorim path alternativ.")
        # Provë e dytë: mbase config është brenda yolo-live/config
        alt_p = BASE_DIR.parent / "config" / "config.yaml"
        if alt_p.exists():
            p = alt_p
        else:
            return {"camera_source": 0, "backend": "opencv", "confidence": 0.25}
    
    with open(p, "r") as f:
        return yaml.safe_load(f)

def create_app():
    app = FastAPI(title="yolo-live")

    cfg = load_config()
    
    # Sigurohemi që model_loader të mos bllokojë nisjen e app
    try:
        model_loader = ModelLoader(cfg)
    except Exception as e:
        print(f"Gabim kritik në ModelLoader: {e}")
        model_loader = None

    app.include_router(api_router, prefix="/api")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/")
    async def viewer():
        # Templates janë në src/templates/viewer.html
        template_path = BASE_DIR / "templates" / "viewer.html"
        if not template_path.exists():
            return JSONResponse({
                "error": "Template mungon", 
                "path_i_kërkuar": str(template_path)
            }, status_code=404)
        return FileResponse(template_path)

    def gen_frames():
        cam_src = cfg.get("camera_source", 0)
        # Përdorim generatorin që rregulluam për Codespaces
        for frame in camera_generator(cam_src):
            if model_loader:
                try:
                    detections = model_loader.predict(frame)
                    frame = model_loader.draw(frame, detections)
                except Exception:
                    pass # Vazhdo me frame origjinal nëse AI dështon
            
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    @app.get("/stream")
    def stream():
        return StreamingResponse(
            gen_frames(), 
            media_type="multipart/x-mixed-replace; boundary=frame"
        )

    return app