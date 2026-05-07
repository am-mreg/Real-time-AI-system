import cv2
import numpy as np
import time

def camera_generator(source=0):
    # Detyrojmë dështimin e kamerës në Codespace për të kaluar te testimi
    cap = cv2.VideoCapture(source)
    
    if False: # E bëjmë False që të kalojë direkt te kodi sintetik poshtë
        pass
    else:
        print("Kamera nuk u gjet. Duke gjeneruar video artificiale...")
        frame_count = 0
        while True:
            # 1. Krijo një imazh me ngjyrë blu të errët (BGR format)
            # Madhësia 480x640 është standarde dhe hapet lehtë në browser
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (40, 20, 20) # Ngjyrë e errët

            # 2. Shto një rrjetë (grid) që të duket sikur po skanohet
            for i in range(0, 640, 100):
                cv2.line(frame, (i, 0), (i, 480), (50, 50, 50), 1)
            for i in range(0, 480, 80):
                cv2.line(frame, (0, i), (640, i), (50, 50, 50), 1)

            # 3. Vizato katrorin që lëviz
            x = (frame_count * 10) % 600
            cv2.rectangle(frame, (x, 180), (x + 80, 260), (0, 255, 0), -1)
            cv2.putText(frame, "SISTEMI LIVE - TEST", (x, 170), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 4. Shto Timestamp-in
            current_time = time.strftime("%H:%M:%S")
            cv2.putText(frame, f"Koha: {current_time}", (20, 450), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            yield frame
            frame_count += 1
            time.sleep(0.04) # 25 FPS