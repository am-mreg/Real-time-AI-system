import asyncio
import base64
import cv2
import numpy as np
from fastapi import WebSocket

async def process_frames_via_ws(websocket: WebSocket, model_loader=None):
    await websocket.accept()
    try:
        while True:
            # 1. Merr të dhënat nga Browseri (Frontend dërgon frame-in)
            data = await websocket.receive_json()
            b64_data = data.get("image")
            
            if b64_data:
                # Hiq header-in e base64 nëse ekziston (p.sh., "data:image/jpeg;base64,")
                if "," in b64_data:
                    b64_data = b64_data.split(",")[1]
                
                # 2. Dekodo imazhin nga base64 në format që e kupton OpenCV
                img_bytes = base64.b64decode(b64_data)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                # 3. Bëj detektimin me modelin tënd AI
                detections = model_loader.predict(frame) if model_loader else []
                out = model_loader.draw(frame, detections) if model_loader else frame
                
                # 4. Kthe imazhin e përpunuar sërish në base64
                _, buffer = cv2.imencode(".jpg", out)
                out_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
                
                # 5. Dërgo imazhin e ri mbrapsht në frontend
                await websocket.send_json({"image": out_b64})
                
    except Exception as e:
        print(f"WebSocket u shkëput ose pati error: {e}")
    finally:
        await websocket.close()