import cv2, numpy as np

def xywh_to_xyxy(x, y, w, h):
    x1 = int(x - w/2)
        y1 = int(y - h/2)
            x2 = int(x + w/2)
                y2 = int(y + h/2)
                    return x1, y1, x2, y2

                    def non_max_suppression(boxes, scores, iou_threshold=0.45):
                        # simple wrapper to cv2.dnn.NMSBoxes if needed
                            import cv2
                                idxs = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_threshold)
                                    if len(idxs) == 0:
                                            return []
                                                return [int(i[0]) if isinstance(i, (list, tuple, np.ndarray)) else int(i) for i in idxs]