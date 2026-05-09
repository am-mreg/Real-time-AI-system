"""
Backwards-compatible wrapper for ModelLoader
File: src/detector/detector.py
"""

from typing import Any, Optional

# Try relative import first, then top-level
try:
    from src.detector.model_loader import ModelLoader  # type: ignore
except Exception:
    try:
        from detector.model_loader import ModelLoader  # type: ignore
    except Exception:
        ModelLoader = None  # type: ignore


class Detector:
    """
    Simple wrapper around ModelLoader to provide a stable API:
      - Detector(cfg) -> constructs underlying ModelLoader if available
      - predict(frame) -> returns detections or None on error
      - draw(frame, detections) -> returns annotated frame or original frame on error
    """

    def __init__(self, cfg: dict):
        self.loader: Optional[Any] = None
        if ModelLoader is None:
            # ModelLoader not available; keep loader as None
            return

        try:
            self.loader = ModelLoader(cfg)
        except Exception as e:
            # Fail gracefully; user code can still call Detector but methods will be no-ops
            # Logging is intentionally omitted here to keep this module dependency-free;
            # the application can log if desired.
            self.loader = None

    def predict(self, frame: Any) -> Optional[Any]:
        """
        Run prediction on a single frame. Returns detections or None on failure.
        """
        if self.loader is None:
            return None
        try:
            return self.loader.predict(frame)
        except Exception:
            return None

    def draw(self, frame: Any, detections: Any) -> Any:
        """
        Draw detections on the frame. Returns the annotated frame or the original frame on failure.
        """
        if self.loader is None:
            return frame
        try:
            return self.loader.draw(frame, detections)
        except Exception:
            return frame
