import cv2
from enum import Enum
from datetime import datetime
import ffmpeg
import logging
from pathlib import Path
import threading
import time
import json
from collections import deque
import asyncio

import utils
from motion_detection import MotionDetector
from object_detection import ObjectDetector
from video_utils import VideoLoggerHandler

class DetectionManager():
    def __init__(self, camera_feed):
        self.video_logger_handler = VideoLoggerHandler()

        self.camera_feed = camera_feed
        self.object_detector = ObjectDetector(self.camera_feed, self.video_logger_handler)
        self.motion_detector = MotionDetector(self.camera_feed, self.video_logger_handler)

    def start(self):
        self.object_detector.start()
        self.motion_detector.start()
    
    def _decide_recording(self):
        while True:
            last_motion_detected_ts = self.motion_detector.last_motion_detection_time
            last_major_motion_detected_ts = self.motion_detector.last_major_motion_detection_time
            last_object_detected_ts = self.object_detector.last_detection_time
            ts = time.time_ns()
            if self.camera_feed.is_recording:
                if ts-last_object_detected_ts > 10 and ts-last_motion_detected_ts > 5:
                    self.camera_feed.stop_recording()
                time.sleep(3)
            else:
                if ts-last_object_detected_ts < 10 and ts-last_major_motion_detected_ts < 5:
                    self._start_recording()
                time.sleep(0.5)
    
    def _start_recording(self):
        video_id = datetime.now().strftime(utils.DATETIME_FORMAT)
        self.camera_feed.start_recording(video_id)
        self.video_logger_handler.create_logger(video_id)
        self.object_detector.set_is_logging(True)
        self.motion_detector.set_is_logging(True)

    def _stop_recording(self):
        self.camera_feed.stop_recording()
        self.object_detector.set_is_logging(False)
        self.motion_detector.set_is_logging(False)
        self.video_logger_handler.close_logger()


    def stop(self):
        self.motion_detector.is_running = False
        self.object_detector.is_running = False
        if self.camera_feed.is_recording:
            self._stop_recording()
