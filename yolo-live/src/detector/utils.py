"""
Utility helpers for detector module.

Provides:
- xywh_to_xyxy: convert center-x, center-y, width, height -> x1,y1,x2,y2
- non_max_suppression: wrapper around OpenCV NMS or a numpy fallback

Functions accept scalars or numpy arrays/lists.
"""
from typing import List, Sequence, Tuple, Union, Optional
import numpy as np
import cv2

Number = Union[int, float]
Box = Tuple[int, int, int, int]  # x1, y1, x2, y2


def xywh_to_xyxy(x: Number, y: Number, w: Number, h: Number) -> Box:
    """
    Convert center x,y and width,height to top-left/bottom-right coordinates.

    Returns integers (x1, y1, x2, y2).
    """
    x1 = int(round(x - w / 2.0))
    y1 = int(round(y - h / 2.0))
    x2 = int(round(x + w / 2.0))
    y2 = int(round(y + h / 2.0))
    return x1, y1, x2, y2


def _to_xyxy_array(boxes: Sequence[Sequence[Number]]) -> np.ndarray:
    """
    Normalize boxes to numpy array in XYXY format.
    Accepts boxes in either [x, y, w, h] (center or top-left ambiguous) or [x1,y1,x2,y2].
    Heuristic: if width <= x2-x1 it's treated as xywh when len==4 and x2 > x1 etc.
    For safety, if values look like xywh (w,h small relative to coords) convert using center assumption.
    """
    arr = np.asarray(boxes, dtype=float)
    if arr.size == 0:
        return arr.reshape((0, 4))

    if arr.shape[1] != 4:
        raise ValueError("Each box must have 4 elements")

    # Heuristic: if second pair is greater than first pair -> likely x1,y1,x2,y2
    # If any x2 <= x1 or y2 <= y1, treat as center-x, center-y, w, h
    x1, y1, x2, y2 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    if np.all((x2 > x1) & (y2 > y1)):
        # Already xyxy
        out = np.stack([x1, y1, x2, y2], axis=1)
    else:
        # Treat as center x,y,w,h
        cx, cy, w, h = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        out = np.stack([x1, y1, x2, y2], axis=1)

    return out


def non_max_suppression(
    boxes: Sequence[Sequence[Number]],
    scores: Sequence[Number],
    iou_threshold: float = 0.45,
    score_threshold: float = 0.0,
) -> List[int]:
    """
    Perform Non-Maximum Suppression.

    Parameters
    - boxes: sequence of boxes. Each box can be [x1,y1,x2,y2] or [cx,cy,w,h].
    - scores: sequence of confidence scores (same length as boxes).
    - iou_threshold: IoU threshold for suppression.
    - score_threshold: minimum score to keep a box.

    Returns list of kept indices (ints) relative to the input order.
    """
    if boxes is None or len(boxes) == 0:
        return []

    boxes_arr = _to_xyxy_array(boxes)
    scores_arr = np.asarray(scores, dtype=float)

    if boxes_arr.shape[0] != scores_arr.shape[0]:
        raise ValueError("boxes and scores must have the same length")

    # Filter by score_threshold
    keep_mask = scores_arr > score_threshold
    if not np.any(keep_mask):
        return []

    boxes_arr = boxes_arr[keep_mask]
    scores_arr = scores_arr[keep_mask]
    original_indices = np.nonzero(keep_mask)[0].tolist()

    # Try OpenCV NMS first (cv2.dnn.NMSBoxes expects boxes in [x,y,w,h] with top-left)
    try:
        # Convert xyxy -> x,y,w,h (top-left)
        xywh = np.zeros_like(boxes_arr)
        xywh[:, 0] = boxes_arr[:, 0]
        xywh[:, 1] = boxes_arr[:, 1]
        xywh[:, 2] = boxes_arr[:, 2] - boxes_arr[:, 0]
        xywh[:, 3] = boxes_arr[:, 3] - boxes_arr[:, 1]

        # cv2.dnn.NMSBoxes expects Python lists
        boxes_list = xywh.tolist()
        scores_list = scores_arr.tolist()

        # Note: cv2.dnn.NMSBoxes returns a list/array of indices (possibly nested)
        idxs = cv2.dnn.NMSBoxes(boxes_list, scores_list, score_threshold, iou_threshold)

        if idxs is None or len(idxs) == 0:
            return []

        # Normalize indices to flat list of ints
        flat = []
        for i in idxs:
            if isinstance(i, (list, tuple, np.ndarray)):
                flat.append(int(i[0]))
            else:
                flat.append(int(i))
        # Map back to original indices
        return [original_indices[i] for i in flat]
    except Exception:
        # Fallback: pure numpy NMS (standard implementation)
        x1 = boxes_arr[:, 0]
        y1 = boxes_arr[:, 1]
        x2 = boxes_arr[:, 2]
        y2 = boxes_arr[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores_arr.argsort()[::-1]

        keep: List[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        # Map kept indices back to original indices
        return [original_indices[int(k)] for k in keep]
