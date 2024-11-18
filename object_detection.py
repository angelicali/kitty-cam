from ultralytics import YOLO
import json

class ObjectDetector():
    def __init__(self):
        self.model = YOLO("finetuned_ncnn_model")
    
    def detect(self, frame, track=True, persist=True, threshold=0.1):
        if track:
            results = self.model.track(frame, persist=persist)[0]
        else:
            results = self.model(frame)[0]
        objects = json.loads(results.to_json())
        return [obj for obj in objects if obj['confidence'] >= threshold]

    def detect_anything(self, frame, threshold=0.1):
        detections = self.detect(frame, threshold)
        return len(detections)>0


        
