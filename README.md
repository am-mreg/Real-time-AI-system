# yolo-live

Live object detection (camera/RTSP) using YOLO (Ultralytics) or OpenCV DNN.
Production-ready repo with FastAPI, Docker, CI, and tests.

## Features
- Live camera/RTSP stream processing
- Backend switch: **yolov8** (Ultralytics) or **opencv** (OpenCV DNN)
- FastAPI endpoints for video stream and health
- Dockerfile and docker-compose for containerized deployment
- CI pipeline for linting, tests, and Docker build
- Minimal unit tests and utilities for downloading weights

## Requirements
- Python 3.10+
- GPU optional (PyTorch + CUDA) for YOLO; CPU fallback supported
- Recommended: 8+ GB RAM

## Quickstart (local)
1. Create virtualenv:
   python -m venv .venv
      source .venv/bin/activate
      2. Install:
         pip install -r requirements.txt
         3. Download model weights:
            python scripts/download_weights.py --backend yolov8 --model yolov8n.pt
            4. Run:
               uvicorn src.app:app --host 0.0.0.0 --port 8000

               ## Docker
               Build:
                 docker build -t yolo-live:latest .
                 Run:
                   docker run --device /dev/video0 -p 8000:8000 -v $(pwd)/weights:/app/weights yolo-live:latest

                   ## Config
                   Edit `config/config.yaml` to change backend, model, device, confidence, and camera source.

                   ## Tests
                     pytest -q

                     ## Contributing
                     Follow standard GitHub flow. Do not commit model weights.

                     ## License
                     MIT

