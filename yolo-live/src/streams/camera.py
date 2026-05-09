"""
src/stream/camera.py

camera_generator(source=0)

Tries to open a real camera using OpenCV VideoCapture. If the camera cannot be opened
(e.g., running inside Codespaces or no device available), it falls back to a synthetic
test video stream (480x640 BGR frames) that is safe for development and browser preview.

Yields:
    frame (np.ndarray): BGR image suitable for encoding with cv2.imencode.
"""
import cv2
import numpy as np
import time
import logging
from typing import Generator, Union

logger = logging.getLogger("camera")
logger.setLevel(logging.INFO)


def _open_capture(source: Union[int, str]) -> Union[cv2.VideoCapture, None]:
    """Try to open VideoCapture for numeric or string source. Return None on failure."""
    try:
        # If source looks like an integer index, convert
        if isinstance(source, str) and source.isdigit():
            src = int(source)
        else:
            src = source
        cap = cv2.VideoCapture(src, cv2.CAP_ANY)
        # Small delay to allow camera to initialize
        time.sleep(0.1)
        if cap is not None and cap.isOpened():
            logger.info(f"Opened camera source: {src}")
            return cap
        else:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            logger.info(f"Could not open camera source: {src}")
            return None
    except Exception as e:
        logger.exception(f"Exception while opening capture: {e}")
        return None


def camera_generator(source: Union[int, str] = 0, fps: float = 25.0) -> Generator[np.ndarray, None, None]:
    """
    Generator that yields BGR frames.

    Parameters:
    - source: camera index (0,1,...) or device path ("/dev/video0") or stream URL.
    - fps: target frames per second for synthetic fallback or when reading fails.

    Behavior:
    - If a real camera can be opened, yields frames from it.
    - Otherwise yields a synthetic animated frame for testing.
    """
    cap = _open_capture(source)

    if cap is not None:
        try:
            # Read frames continuously from the real camera
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    # brief sleep to avoid busy loop if camera temporarily fails
                    time.sleep(0.01)
                    continue
                yield frame
        except GeneratorExit:
            # consumer closed generator
            pass
        except Exception:
            logger.exception("Error while reading from camera.")
        finally:
            try:
                cap.release()
            except Exception:
                pass
    else:
        # Synthetic fallback stream
        logger.info("No camera available; using synthetic fallback stream.")
        frame_count = 0
        height, width = 480, 640
        period = 1.0 / max(1.0, float(fps))

        try:
            while True:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                # dark bluish background (BGR)
                frame[:] = (40, 20, 20)

                # grid overlay
                for i in range(0, width, 100):
                    cv2.line(frame, (i, 0), (i, height), (50, 50, 50), 1)
                for j in range(0, height, 80):
                    cv2.line(frame, (0, j), (width, j), (50, 50, 50), 1)

                # moving square
                x = (frame_count * 10) % (width + 100) - 50
                x1 = max(0, x)
                x2 = min(width - 1, x + 80)
                cv2.rectangle(frame, (x1, 180), (x2, 260), (0, 255, 0), -1)
                cv2.putText(frame, "SISTEMI LIVE - TEST", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # timestamp
                current_time = time.strftime("%H:%M:%S")
                cv2.putText(frame, f"Koha: {current_time}", (20, height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                yield frame
                frame_count += 1
                time.sleep(period)
        except GeneratorExit:
            pass
        except Exception:
            logger.exception("Error in synthetic camera generator.")
