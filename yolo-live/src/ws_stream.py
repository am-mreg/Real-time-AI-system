import asyncio
import base64
import logging
from typing import Optional

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("ws_stream")
logger.setLevel(logging.INFO)


async def process_frames_via_ws(websocket: WebSocket, model_loader: Optional[object] = None) -> None:
    """
    Accept a WebSocket connection and process incoming frames.

    Parameters
    - websocket: FastAPI WebSocket instance
    - model_loader: optional object exposing .predict(frame) -> detections
                    and .draw(frame, detections) -> annotated_frame
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted.")
    try:
        while True:
            try:
                msg = await websocket.receive_text()
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected by client.")
                break
            except Exception as e:
                # Try receive_json as fallback (some clients send JSON)
                try:
                    data = await websocket.receive_json()
                except Exception as e2:
                    logger.exception("Failed to receive message from websocket.", exc_info=True)
                    await asyncio.sleep(0.1)
                    continue
                else:
                    msg = None
                    data_in = data

            # If we got raw text, try to parse as JSON-like structure
            if msg is not None:
                try:
                    import json
                    data_in = json.loads(msg)
                except Exception:
                    # If it's not JSON, skip
                    logger.warning("Received non-JSON text over websocket; ignoring.")
                    await asyncio.sleep(0.01)
                    continue

            # Expect a dict with key "image"
            if not isinstance(data_in, dict):
                logger.warning("Received websocket payload is not a JSON object; ignoring.")
                await asyncio.sleep(0.01)
                continue

            b64_data = data_in.get("image")
            if not b64_data:
                # Nothing to do
                await asyncio.sleep(0.01)
                continue

            # Remove data URL prefix if present
            if isinstance(b64_data, str) and "," in b64_data and b64_data.startswith("data:"):
                b64_data = b64_data.split(",", 1)[1]

            try:
                img_bytes = base64.b64decode(b64_data)
                np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None:
                    logger.warning("Decoded frame is None; skipping.")
                    await asyncio.sleep(0.01)
                    continue
            except Exception as e:
                logger.exception("Failed to decode incoming base64 image.", exc_info=True)
                await asyncio.sleep(0.01)
                continue

            # Run model inference if available
            try:
                detections = model_loader.predict(frame) if (model_loader is not None and hasattr(model_loader, "predict")) else []
            except Exception:
                logger.exception("Error during model prediction; continuing with original frame.")
                detections = []

            try:
                out_frame = model_loader.draw(frame, detections) if (model_loader is not None and hasattr(model_loader, "draw")) else frame
            except Exception:
                logger.exception("Error drawing detections; sending original frame.")
                out_frame = frame

            # Encode back to JPEG and base64
            try:
                success, buffer = cv2.imencode(".jpg", out_frame)
                if not success:
                    logger.warning("cv2.imencode failed; skipping send.")
                    await asyncio.sleep(0.01)
                    continue
                out_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
                # Send JSON with image (without data: prefix to keep payload smaller)
                await websocket.send_json({"image": out_b64})
            except Exception:
                logger.exception("Failed to encode/send processed frame.", exc_info=True)
                await asyncio.sleep(0.01)
                continue

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception:
        logger.exception("Unexpected error in WebSocket processing loop.")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("WebSocket connection closed.")