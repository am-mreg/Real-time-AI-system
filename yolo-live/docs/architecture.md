# Architecture

## Components
- **FastAPI app** (`src/app.py`) exposes:
  - `/stream` multipart MJPEG stream
    - `/api/reload` reload config
      - `/health` health check
      - **ModelLoader** (`src/detector/model_loader.py`) abstracts backend differences:
        - `yolov8` via Ultralytics
          - `opencv` via OpenCV DNN (ONNX)
          - **Camera stream** (`src/streams/camera.py`) handles webcam/RTSP
          - **Scripts** for downloading weights and CI for tests/build

          ## Deployment
          - Containerize with Docker; mount `weights/` at runtime.
          - For GPU, use appropriate base image and pass devices.

          ## Security & Production Notes
          - Do not commit weights.
          - Validate model checksums before loading.
          - Use process manager (systemd, k8s) for production.