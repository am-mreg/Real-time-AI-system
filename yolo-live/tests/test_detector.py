import numpy as np
from src.detector.model_loader import ModelLoader
import pytest
import os

@pytest.fixture
def cfg(tmp_path):
    # create a dummy config that points to a non-existent model for opencv
        return {
                "backend": "opencv",
                        "model": "weights/dummy.onnx",
                                "confidence": 0.25,
                                        "device": -1,
                                                "imgsz": 320
                                                    }

                                                    def test_model_loader_missing_model(cfg):
                                                        with pytest.raises(FileNotFoundError):
                                                                ModelLoader(cfg)

                                                                def test_camera_generator_no_device(monkeypatch):
                                                                    from src.streams.camera import camera_generator
                                                                        class DummyCap:
                                                                                def isOpened(self): return False
                                                                                    monkeypatch.setattr("cv2.VideoCapture", lambda src: DummyCap())
                                                                                        with pytest.raises(RuntimeError):
                                                                                                list(camera_generator(0))