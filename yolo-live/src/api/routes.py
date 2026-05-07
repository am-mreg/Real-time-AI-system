from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import yaml, os
from src.detector.model_loader import ModelLoader

router = APIRouter()

class ReloadRequest(BaseModel):
    config_path: str = None

@router.post("/reload")
def reload_model(req: ReloadRequest):
    cfg_path = req.config_path or os.environ.get("CONFIG_PATH", "config/config.yaml")
    if not os.path.exists(cfg_path):
        raise HTTPException(status_code=400, detail="config not found")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    # instantiate new loader
    loader = ModelLoader(cfg)
    # replace global instance if desired (simple approach)
    # NOTE: in production use dependency injection or app.state
    return {"status":"reloaded", "backend": cfg.get("backend")}