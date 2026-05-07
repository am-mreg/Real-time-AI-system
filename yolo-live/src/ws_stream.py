import asyncio
import base64
import cv2
import numpy as np
from fastapi import WebSocket

async def send_frames_via_ws(websocket: WebSocket, camera_source=0, model_loader=None):
    await websocket.accept()
        cap = cv2.VideoCapture(camera_source)
            if not cap.isOpened():
                    await websocket.close(code=1001)
                            return
                                try:
                                        while True:
                                                    ret, frame = cap.read()
                                                                if not ret:
                                                                                break
                                                                                            # run detection if loader provided
                                                                                                        detections = model_loader.predict(frame) if model_loader else []
                                                                                                                    out = model_loader.draw(frame, detections) if model_loader else frame
                                                                                                                                _, buffer = cv2.imencode(".jpg", out)
                                                                                                                                            jpg_bytes = buffer.tobytes()
                                                                                                                                                        b64 = base64.b64encode(jpg_bytes).decode("utf-8")
                                                                                                                                                                    await websocket.send_json({"image": b64})
                                                                                                                                                                                await asyncio.sleep(0.03)  # throttle ~30fps
                                                                                                                                                                                    except Exception:
                                                                                                                                                                                            pass
                                                                                                                                                                                                finally:
                                                                                                                                                                                                        cap.release()
                                                                                                                                                                                                                await websocket.close()