from ultralytics import YOLO
import json

class ObjectDetector():
    def __init__(self):
        self.model = YOLO("finetuned_ncnn_model")
    
    def detect(self, frame):
        results = self.model(frame)[0]
        return json.loads(results.to_json())

    def detect_anything(self, frame):
        detections = self.detect(frame)
        for obj in detections:
            if obj['confidence'] >= 0.1:
                return True
        return False


        