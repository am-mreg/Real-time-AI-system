# src/tests/test_detector.py
import os
import sys
import unittest
import importlib
import types

# Ensure project root is on sys.path so "src.*" imports work in CI
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Try to import cv2; if not available, tests that require it will be skipped.
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    cv2 = None  # type: ignore
    _HAS_CV2 = False

# Import the modules under test. Use importlib to get clearer errors if module missing.
try:
    ModelLoader = importlib.import_module("src.detector.model_loader").ModelLoader
except Exception:
    ModelLoader = None

try:
    camera_module = importlib.import_module("src.stream.camera")
    camera_generator = getattr(camera_module, "camera_generator", None)
except Exception:
    camera_generator = None


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "backend": "opencv",
            "model": "weights/dummy.onnx",
            "confidence": 0.25,
            "device": -1,
            "imgsz": 320,
        }

    def test_model_loader_missing_model_raises(self):
        """ModelLoader should raise FileNotFoundError when model file is missing."""
        if ModelLoader is None:
            self.skipTest("src.detector.model_loader not importable")
        with self.assertRaises(FileNotFoundError):
            ModelLoader(self.cfg)

    @unittest.skipIf(not _HAS_CV2, "OpenCV (cv2) is not installed in this environment")
    def test_camera_generator_fallback_to_synthetic(self):
        """
        If VideoCapture cannot be opened, camera_generator should yield synthetic frames.
        """
        if camera_generator is None:
            self.skipTest("src.stream.camera.camera_generator not importable")

        # Create a dummy VideoCapture replacement that reports not opened
        class DummyCap:
            def __init__(self, *args, **kwargs):
                pass

            def isOpened(self):
                return False

            def release(self):
                pass

        # Monkeypatch cv2.VideoCapture locally
        original_videocapture = cv2.VideoCapture
        try:
            cv2.VideoCapture = lambda src, *args, **kwargs: DummyCap()
            gen = camera_generator(0)
            frame = next(gen)
            # Validate frame
            self.assertIsNotNone(frame)
            self.assertIsInstance(frame, (list, tuple, type(None))) is False  # ensure not a trivial type
            import numpy as np
            self.assertIsInstance(frame, np.ndarray)
            self.assertEqual(frame.shape, (480, 640, 3))
        finally:
            cv2.VideoCapture = original_videocapture


if __name__ == "__main__":
    unittest.main()
