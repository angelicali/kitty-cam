from ultralytics import YOLO
import json
import multiprocessing
from collections import deque
import time

class ObjectDetector():
    def __init__(self, camera_feed):
        self.camera_feed = camera_feed
        self.is_running = multiprocessing.Value('i', 1)
        self.results_queue = multiprocessing.Queue()
        self.last_detection_time = multiprocessing.Value('i', 0)
    
    def start(self):
        self.process = multiprocessing.Process(target=self._loop_detection, args=(self.camera_feed.frame_queue, self.camera_feed.is_recording, self.last_detection_time, self.is_running, self.results_queue))
        self.process.daemon = True
        self.process.start()

    def cleanup(self):
        print("stopping object detector...")
        self.is_running.value = 0
        self.process.join()
        print("object detector process joined")

    def _loop_detection(self, frame_queue, is_recording, last_detection_time, is_running, detection_results_queue):
        model = YOLO("finetuned_ncnn_model")
        while is_running.value == 1:
            frame = frame_queue.get()
            if frame is None:
                break
            ts = int(time.time())
            results = model.track(frame, persist=True)[0]
            objects = json.loads(results.to_json())
            if len(objects) > 0:
                last_detection_time.value = ts
            if not is_recording.value:
                time.sleep(2)
            else:
                time.sleep(1)
                detection_results_queue.put((ts,objects))
