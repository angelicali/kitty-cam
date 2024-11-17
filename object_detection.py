from ultralytics import YOLO
import json

class ObjectDetector():
    def __init__(self):
        self.model = YOLO("finetuned_ncnn_model")
    
    def detect(self, frame, threshold=0.1):
        results = self.model(frame)[0]
        objects = json.loads(results.to_json())
        return [obj for obj in objects if obj['confidence'] >= threshold]

    def detect_anything(self, frame, threshold=0.1):
        detections = self.detect(frame, threshold)
        return len(detections)>0


        
