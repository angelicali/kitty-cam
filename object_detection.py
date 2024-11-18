from ultralytics import YOLO
import json
import multiprocessing
from collections import deque
import time

class ObjectDetector():
    def __init__(self, camera_feed, video_logger_handler):
        self.model = YOLO("finetuned_ncnn_model")
        self.camera_feed = camera_feed
        self.is_running = True
        self.results_queue = deque(maxlan=30)
        self.last_detection_time = 0
        self.video_logger_handler = video_logger_handler
        self.is_logging = False
    
    def start(self):
        self.process = multiprocessing.Process(target=self._loop_detection)
        self.process.daemon = True
        self.process.start()

    def cleanup(self):
        self.process.join()

    def _loop_detection(self):
        while self.is_running:
            ts = time.time_ns()
            results = self.detect(self.camera_feed.get_frame(), threshold=0.0)
            self.results_queue.append((ts,results))
            if len(results) > 0:
                self.last_detection_time = ts
                if self.is_logging:
                    self.video_logger_handler((ts, results))

    def set_is_logging(self, is_logging):
        self.is_logging = is_logging

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

        
