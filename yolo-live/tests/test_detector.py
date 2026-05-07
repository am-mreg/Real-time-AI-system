import numpy as np
import pytest
import cv2
from src.detector.model_loader import ModelLoader
from src.streams.camera import camera_generator

@pytest.fixture
def cfg():
    """Krijon një konfigurim dummy për testim."""
    return {
        "backend": "opencv",
        "model": "weights/dummy.onnx",
        "confidence": 0.25,
        "device": -1,
        "imgsz": 320
    }

def test_model_loader_missing_model(cfg):
    """Verifikon që ModelLoader ngre FileNotFoundError nëse modeli mungon."""
    with pytest.raises(FileNotFoundError):
        ModelLoader(cfg)

def test_camera_generator_fallback_to_synthetic(monkeypatch):
    """
    Verifikon që camera_generator kalon në 'synthetic mode' 
    nëse kamera nuk hapet (në vend që të japë RuntimeError).
    """
    # Krijojmë një Mock për VideoCapture që kthen isOpened() = False
    class DummyCap:
        def isOpened(self): return False
        def release(self): pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda src: DummyCap())

    # Marrim framin e parë nga generatori
    gen = camera_generator(0)
    frame = next(gen)

    # Verifikojmë që kemi marrë një imazh (numpy array) dhe jo një error
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (480, 640, 3)