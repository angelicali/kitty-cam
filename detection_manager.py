import cv2
from datetime import datetime
import time
import threading

import utils
from motion_detection import MotionDetector
from object_detection import ObjectDetector
from video_utils import VideoLoggerHandler

class DetectionManager():
    def __init__(self, camera_feed):
        self.video_logger_handler = VideoLoggerHandler()
        self.camera_feed = camera_feed
        self.object_detector = ObjectDetector(self.camera_feed)
        self.motion_detector = MotionDetector(self.camera_feed, self.video_logger_handler)
        self.is_running = False

    def start(self):
        self.object_detector.start()
        self.motion_detector.start()
        self.is_running = True
        self.thread = threading.Thread(target=self._decide_recording)
        self.thread.daemon = True
        self.thread.start()

    def _decide_recording(self):
        while self.is_running:
            last_motion_detected_ts = self.motion_detector.last_motion_detection_time
            last_major_motion_detected_ts = self.motion_detector.last_major_motion_detection_time
            last_object_detected_ts = self.object_detector.last_detection_time.value
            ts = time.time()
            if self.camera_feed.get_is_recording():
                while not self.object_detector.results_queue.empty():
                    objd_results = self.object_detector.results_queue.get()
                    self.video_logger_handler.log(objd_results)
                if ts-last_object_detected_ts > 10 and ts-last_motion_detected_ts > 5:
                    self.camera_feed.stop_recording()
                time.sleep(3)
            else:
                if ts-last_object_detected_ts < 10 and ts-last_major_motion_detected_ts < 5:
                    self._start_recording()
                time.sleep(0.5)
    
    def _start_recording(self):
        video_id = datetime.now().strftime(utils.DATETIME_FORMAT)
        self.video_logger_handler.create_logger(video_id)
        self.camera_feed.start_recording(video_id)

    def _stop_recording(self):
        self.camera_feed.stop_recording()
        self.video_logger_handler.close_logger()


    def stop(self):
        print("stopping detection manager...")
        self.is_running = False
        self.thread.join()
        self.object_detector.cleanup()
        self.motion_detector.cleanup()
        if self.camera_feed.get_is_recording():
            self._stop_recording()
        print("detection manager stopped")

