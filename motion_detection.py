import cv2
import numpy as np
from collections import deque
import threading
import time

class MotionDetector():
    def __init__(self, camera_feed, blur_size=21, threshold=25, min_area=500):
        self.blur_size = blur_size
        self.threshold = threshold
        self.min_area = min_area
        
        self.camera_feed = camera_feed
        self.video_logger = None
        self.is_running = False

        self.prev_frame_blurred = self._blur(self.camera_feed.get_frame())
        self.last_major_motion_detection_time = 0
        self.last_motion_detection_time = 0
        self.results_queue = deque(maxlan=30)

    def start(self):
        self.is_running = True
        self.thread = threading.Process(target=self._loop_detection)
        self.thread.daemon = True
        self.thread.start()

    def _loop_detection(self):
        while self.is_running:
            ts = time.time_ns()
            results = self.detect(self.camera_feed.get_frame())
            self.results_queue.append((ts, results))
            if results['contour_area_max'] >= 500:
                self.last_major_motion_detection_time = ts
                self.last_motion_detection_time = ts
            elif results['contour_area_max'] >= 100:
                self.last_motion_detection_time = ts
            if self.is_logging:
                self.video_logger_handler((ts, results))
            time.sleep(1)

    def set_is_logging(self, is_logging):
        self.is_logging = is_logging

    def _blur(self, frame):
        frame = frame.copy()
        # TODO: is the original frame modified?
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
    
    def detect(self, frame):
        blurred_frame = self._blur(frame)
        raw_delta = cv2.absdiff(self.prev_frame_blurred, blurred_frame)
        self.prev_frame_blurred = blurred_frame

        threshold_delta = cv2.threshold(raw_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        dilated_delta = cv2.dilate(threshold_delta, None, iterations=2)
        contours, _ = cv2.findContours(dilated_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        metrics = {
            'raw_delta_mean_change': np.mean(raw_delta),
            'raw_delta_max_change': float(np.max(raw_delta)),
            'raw_delta_percent_change': (raw_delta > self.threshold).mean() * 100,
            'contour_area_total': sum(cv2.contourArea(c) for c in contours),
            'contour_count': len(contours),
            'contour_area_max': np.max(cv2.contourArea(c) for c in contours)
        }

        return metrics

    def set_blur_size(self, blur_size):
        self.blur_size = blur_size

    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_min_area(self, min_area):
        self.min_area = min_area
    
    def get_configs(self):
        return {"blur_size": self.blur_size, "threshold": self.threshold, "min_area": self.min_area}
