import os, cv2, numpy as np
from typing import Dict, Any

class ModelLoader:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.backend = cfg.get("backend", "yolov8")
        self.device = cfg.get("device", -1)
        self.confidence = cfg.get("confidence", 0.25)
        self.imgsz = cfg.get("imgsz", 640)
        self.model = None
        if self.backend == "yolov8":
            self._load_yolov8(cfg.get("model"))
        elif self.backend == "opencv":
            self._load_opencv(cfg.get("model"))
        else:
            raise ValueError("Unsupported backend")

    def _load_yolov8(self, model_path):
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError("Ultralytics YOLO not installed") from e
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = YOLO(model_path)

    def _load_opencv(self, model_path):
        # expects ONNX model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = cv2.dnn.readNet(model_path)
        if self.device is not None and self.device >= 0:
            # attempt CUDA backend if available
            try:
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            except Exception:
                pass

    def predict(self, frame):
        if self.backend == "yolov8":
            return self._predict_yolo(frame)
        else:
            return self._predict_opencv(frame)

    def _predict_yolo(self, frame):
        # Ultralytics accepts numpy array BGR
        results = self.model(frame, imgsz=self.imgsz, conf=self.confidence)
        # results is a Results object; convert to list of dicts
        out = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                out.append({"xyxy": xyxy, "conf": conf, "class": cls})
        return out

    def _predict_opencv(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (self.imgsz, self.imgsz), swapRB=True, crop=False)
        self.model.setInput(blob)
        preds = self.model.forward()
        # Parsing depends on model; placeholder returns empty
        return []

    def draw(self, frame, detections):
        import cv2
        for det in detections:
            x1, y1, x2, y2 = map(int, det["xyxy"])
            conf = det.get("conf", 0)
            cls = det.get("class", 0)
            label = f"{cls}:{conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        return frame