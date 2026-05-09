"""
src/detector/model_loader.py

Robust loader for detection models:
- backend "yolov8" uses ultralytics.YOLO
- backend "opencv" uses cv2.dnn (ONNX or other supported net)
"""
from typing import Dict, Any, List
import os
import cv2
import numpy as np
import logging

logger = logging.getLogger("model_loader")
logger.setLevel(logging.INFO)


class ModelLoader:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}
        self.backend = self.cfg.get("backend", "yolov8")
        self.device = self.cfg.get("device", -1)  # -1 means CPU for ultralytics
        self.confidence = float(self.cfg.get("confidence", 0.25))
        self.imgsz = int(self.cfg.get("imgsz", 640))
        self.model_path = self.cfg.get("model", None)
        self.model = None

        if self.backend == "yolov8":
            self._load_yolov8(self.model_path)
        elif self.backend == "opencv":
            self._load_opencv(self.model_path)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _load_yolov8(self, model_path: str):
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            logger.error("Ultralytics YOLO not available. Install `ultralytics` to use yolov8 backend.")
            raise RuntimeError("Ultralytics YOLO not installed") from e

        if not model_path:
            raise ValueError("model path must be provided for yolov8 backend")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        try:
            self.model = YOLO(model_path)
            # If device specified and ultralytics supports .to()
            try:
                if isinstance(self.device, int) and self.device >= 0:
                    # ultralytics uses 'cuda' or 'cpu' strings; map device int -> cuda:0 etc.
                    cuda_str = f"cuda:{self.device}"
                    self.model.to(cuda_str)
                    logger.info(f"YOLO model moved to {cuda_str}")
                else:
                    self.model.to("cpu")
            except Exception:
                # Some ultralytics versions manage device via env or init; ignore if not supported
                logger.debug("Could not set device on YOLO model; continuing with default device.")
        except Exception as e:
            logger.exception(f"Failed to load YOLO model from {model_path}: {e}")
            raise

    def _load_opencv(self, model_path: str):
        if not model_path:
            raise ValueError("model path must be provided for opencv backend")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        try:
            net = cv2.dnn.readNet(model_path)
            # Try to enable CUDA if requested
            if isinstance(self.device, int) and self.device >= 0:
                try:
                    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
                    logger.info("OpenCV DNN configured to use CUDA backend/target.")
                except Exception:
                    logger.warning("CUDA backend not available for OpenCV DNN; using default backend.")
            self.model = net
        except Exception as e:
            logger.exception(f"Failed to load OpenCV model from {model_path}: {e}")
            raise

    def predict(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Predict detections for a single BGR frame.
        Returns list of dicts: {"xyxy":[x1,y1,x2,y2], "conf":float, "class":int}
        """
        if frame is None:
            return []

        if self.backend == "yolov8":
            return self._predict_yolo(frame)
        else:
            return self._predict_opencv(frame)

    def _predict_yolo(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Use ultralytics YOLO model to predict.
        """
        if self.model is None:
            logger.warning("YOLO model not loaded.")
            return []

        try:
            # ultralytics accepts BGR numpy arrays
            results = self.model(frame, imgsz=self.imgsz, conf=self.confidence)
            out = []
            # results may be iterable; each result contains .boxes
            for r in results:
                boxes = getattr(r, "boxes", None)
                if boxes is None:
                    continue
                # boxes.xyxy, boxes.conf, boxes.cls are tensors; iterate safely
                for i in range(len(boxes)):
                    try:
                        # Some ultralytics versions expose arrays differently; handle common cases
                        xyxy = None
                        conf = None
                        cls = None

                        # Try attribute access
                        if hasattr(boxes, "xyxy"):
                            xyxy_val = boxes.xyxy[i]
                            # xyxy_val may be tensor; convert to numpy
                            try:
                                xyxy = xyxy_val.cpu().numpy().tolist()
                            except Exception:
                                xyxy = np.array(xyxy_val).tolist()
                        if hasattr(boxes, "conf"):
                            conf_val = boxes.conf[i]
                            try:
                                conf = float(conf_val.cpu().numpy())
                            except Exception:
                                conf = float(conf_val)
                        if hasattr(boxes, "cls"):
                            cls_val = boxes.cls[i]
                            try:
                                cls = int(cls_val.cpu().numpy())
                            except Exception:
                                cls = int(cls_val)

                        # Fallback if xyxy is nested
                        if xyxy is None:
                            # try boxes.xyxy as array
                            try:
                                xyxy = boxes.xyxy.cpu().numpy()[i].tolist()
                            except Exception:
                                continue

                        out.append({"xyxy": xyxy, "conf": conf or 0.0, "class": cls or 0})
                    except Exception:
                        logger.debug("Skipping a box due to parsing error.", exc_info=True)
                        continue
            return out
        except Exception as e:
            logger.exception(f"Error during YOLO prediction: {e}")
            return []

    def _predict_opencv(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Use OpenCV DNN forward pass. Parsing the output depends on the model architecture.
        This implementation performs a forward pass and returns an empty list by default.
        Extend this method to parse your model's output (for example, YOLOv5/YOLOv8 ONNX outputs).
        """
        if self.model is None:
            logger.warning("OpenCV model not loaded.")
            return []

        try:
            blob = cv2.dnn.blobFromImage(
                frame, scalefactor=1.0 / 255.0, size=(self.imgsz, self.imgsz), mean=(0, 0, 0), swapRB=True, crop=False
            )
            self.model.setInput(blob)
            preds = self.model.forward()

            # NOTE: Parsing preds depends on the model. Many ONNX YOLO models output a (N,85) array
            # where each row is [cx,cy,w,h,conf,class_scores...]. Implement parsing here for your model.
            # For safety, return empty list if we cannot parse.
            detections = []

            # Try a common case: preds shape (1, N, 85) or (N,85)
            arr = np.array(preds)
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]

            if arr.ndim == 2 and arr.shape[1] >= 6:
                # Heuristic parsing: last columns are class scores
                for row in arr:
                    # row: [x, y, w, h, conf, cls0, cls1, ...] or [x1,y1,x2,y2,conf,cls]
                    try:
                        scores = row[5:]
                        if scores.size == 0:
                            conf = float(row[4])
                            cls = int(row[5]) if row.size > 5 else 0
                        else:
                            cls = int(np.argmax(scores))
                            conf = float(scores[cls])
                        # If coordinates are center-based, convert to xyxy
                        x, y, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                        x1 = int(round(x - w / 2.0))
                        y1 = int(round(y - h / 2.0))
                        x2 = int(round(x + w / 2.0))
                        y2 = int(round(y + h / 2.0))
                        detections.append({"xyxy": [x1, y1, x2, y2], "conf": conf, "class": cls})
                    except Exception:
                        continue

            return detections
        except Exception as e:
            logger.exception(f"Error during OpenCV DNN forward: {e}")
            return []

    def draw(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame.
        """
        if frame is None:
            return frame

        try:
            for det in detections or []:
                try:
                    xyxy = det.get("xyxy", None)
                    if not xyxy or len(xyxy) < 4:
                        continue
                    x1, y1, x2, y2 = map(int, xyxy[:4])
                    conf = float(det.get("conf", 0.0))
                    cls = int(det.get("class", 0))
                    label = f"{cls}:{conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                except Exception:
                    continue
        except Exception:
            logger.exception("Error drawing detections on frame.")
        return frame
