"""
yolo-live/api/routes.py

Simple API route to reload model configuration at runtime.

This endpoint reads a YAML config file, instantiates a new ModelLoader,
and replaces the module-level loader reference. In production you should
use FastAPI's dependency injection or app.state to manage shared objects.
"""
from typing import Optional, Any
import os
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import yaml

router = APIRouter()
logger = logging.getLogger("api.routes")

# Module-level loader reference (simple approach for demos/tests)
loader: Optional[Any] = None


class ReloadRequest(BaseModel):
    config_path: Optional[str] = None


@router.post("/reload")
def reload_model(req: ReloadRequest):
    """
    Reload model configuration and instantiate a new ModelLoader.

    Request body (JSON):
      {"config_path": "/path/to/config.yaml"}  # optional

    Returns:
      {"status": "reloaded", "backend": "...", "model": "..."}
    """
    cfg_path = req.config_path or os.environ.get("CONFIG_PATH", "config/config.yaml")

    if not os.path.exists(cfg_path):
        raise HTTPException(status_code=400, detail=f"config not found: {cfg_path}")

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        logger.exception("Failed to read config file")
        raise HTTPException(status_code=500, detail=f"failed to read config: {e}")

    # Import ModelLoader lazily so this module can be imported even if model deps are missing
    try:
        from src.detector.model_loader import ModelLoader
    except Exception as e:
        logger.exception("Failed to import ModelLoader")
        raise HTTPException(status_code=500, detail="ModelLoader not available on this environment")

    # Instantiate the loader and handle common errors (e.g., missing model file)
    try:
        global loader
        loader = ModelLoader(cfg)
    except FileNotFoundError as e:
        logger.exception("Model file not found")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Failed to instantiate ModelLoader")
        raise HTTPException(status_code=500, detail=f"failed to create model loader: {e}")

    return {"status": "reloaded", "backend": cfg.get("backend"), "model": cfg.get("model")}
