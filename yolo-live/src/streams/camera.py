import cv2
from typing import Generator
import numpy as np

def camera_generator(source=0) -> Generator:
    """
    Yields frames (BGR numpy arrays) from a camera or RTSP source.
    For testing in headless environment, generates synthetic frames.
    """
    # Try to open camera first
    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
        finally:
            cap.release()
    else:
        # Generate synthetic frames for testing
        print("Camera not available, generating synthetic frames for testing")
        frame_count = 0
        while True:
            # Create a synthetic frame (640x480 RGB)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add some test content
            cv2.putText(frame, f"Frame {frame_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.rectangle(frame, (100, 100), (200, 200), (0, 255, 0), 2)
            frame_count += 1
            yield frame
            import time
            time.sleep(0.1)  # 10 FPS