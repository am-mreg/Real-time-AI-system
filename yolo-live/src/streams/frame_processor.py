"""
Simple frame processing utilities.

This module exposes `process_frame(frame)` which performs lightweight,
safe preprocessing suitable for feeding into a detector or for streaming:
- validates input
- ensures BGR numpy array
- resizes to a sane max dimension while preserving aspect ratio
- optionally converts color / normalizes if needed (kept minimal here)

Keep this file small and dependency-free so it can be used both in
development (synthetic frames) and production (camera frames).
"""
from typing import Optional
import numpy as np
import cv2

# Maximum size for the longer side to avoid huge frames being streamed
_MAX_SIDE = 1280


def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
    """Ensure frame is a 3-channel BGR uint8 image."""
    if frame is None:
        raise ValueError("frame is None")
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a numpy.ndarray")
    if frame.ndim == 2:
        # single channel -> convert to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        # RGBA -> BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 3:
        # assume already BGR (OpenCV convention)
        pass
    else:
        raise ValueError("Unsupported frame shape: {}".format(frame.shape))
    if frame.dtype != np.uint8:
        # convert to uint8 if possible
        frame = (np.clip(frame, 0, 255)).astype(np.uint8)
    return frame


def _resize_keep_aspect(frame: np.ndarray, max_side: int = _MAX_SIDE) -> np.ndarray:
    """Resize frame so the longer side is <= max_side, preserving aspect ratio."""
    h, w = frame.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return frame
    scale = max_side / float(long_side)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def process_frame(frame: Optional[np.ndarray]) -> np.ndarray:
    """
    Lightweight processing for a single BGR frame.

    - Validates and normalizes input
    - Resizes large frames to a reasonable maximum side length
    - Returns a BGR uint8 numpy array ready for encoding or model input

    Parameters
    - frame: input image (BGR, grayscale, or BGRA) as numpy array

    Returns
    - processed frame (np.ndarray)

    Raises
    - ValueError / TypeError on invalid input
    """
    if frame is None:
        raise ValueError("No frame provided to process_frame")

    # Ensure correct type and channels
    frame = _ensure_bgr(frame)

    # Resize if too large (keeps CPU / network usage reasonable)
    frame = _resize_keep_aspect(frame, _MAX_SIDE)

    # (Optional) additional preprocessing can be added here:
    # - color conversion to RGB for some models: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # - normalization to float32 and scaling to [0,1]
    # - letterbox padding to square shape for certain models
    # Keep this function minimal so it can be reused by different backends.

    return frame
