import cv2
import numpy as np
import time
from typing import Generator

def camera_generator(source=0) -> Generator[np.ndarray, None, None]:
    """
    Yields frames (BGR numpy arrays) from a camera or RTSP source.
    For testing in headless environment (like Codespaces), generates synthetic frames.
    """
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
        # --- ZGJIDHJA PËR CODESPACES ---
        print("Kamera nuk u gjet. Duke gjeneruar frame sintetike për testim...")
        
        frame_count = 0
        # Krijojmë një frame bazë jashtë loop-it për të kursyer memorien (RAM)
        base_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        while True:
            # Kopjojmë frame-in bazë që të mos vizatojmë mbi tekstin e vjetër
            frame = base_frame.copy()
            
            # Shtojmë përmbajtje dinamike
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(frame, f"Test Mode - Frame: {frame_count}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Time: {timestamp}", (50, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
            
            # Vizatojmë një katror që lëviz pak për të parë nëse videoja është "live"
            offset = (frame_count * 5) % 200
            cv2.rectangle(frame, (100 + offset, 150), (200 + offset, 250), (0, 0, 255), 2)
            
            yield frame
            
            frame_count += 1
            time.sleep(0.1)  # Simulo 10 FPS që të mos mbingarkosh CPU-në