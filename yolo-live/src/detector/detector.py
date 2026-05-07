# Backwards-compatible wrapper for ModelLoader
from .model_loader import ModelLoader

class Detector:
    def __init__(self, cfg):
            self.loader = ModelLoader(cfg)

                def predict(self, frame):
                        return self.loader.predict(frame)

                            def draw(self, frame, detections):
                                    return self.loader.draw(frame, detections)