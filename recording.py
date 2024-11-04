""" Recording logic and utils for the camera. """
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

import utils
from motion_detection import MotionDetector

class VideoWriter():
    """ For writing a single video. """
    def __init__(self, filename):
        self.process = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='640X480', r=20.0)
                .output(filename, pix_fmt='yuv420p', vcodec='libx264')
                .overwrite_output()
                .run_async(pipe_stdin=True)
                )
        
    def write(self, frame):
        self.process.stdin.write(frame.tobytes())

    def release(self):
        self.process.stdin.close()
        self.process.wait()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

class VideoLogger():
    def __init__(self, video_id):
        self.logger = logging.getLogger(f"motion_detection_logger_{video_id}")
        self.logger.setLevel(logging.INFO)
        
        self.file_handler = logging.FileHandler(utils.VIDEO_LOG_DIR / f"{video_id}.jsonl", mode='a')
        self.logger.addHandler(self.file_handler)
    
    def log(self, data):
        self.logger.info(json.dumps(data))
    
    def close(self):
        self.logger.removeHandler(self.file_handler)
        self.file_handler.close()

class RecordingState(Enum):
    NOT_RECORDING = 1
    RECORDING = 2
    GRACE_PERIOD = 3 # also recording, but will stop after extended waiting 


""" Main camera recorder, responsible for monitoring and recording decisions. """
class CameraRecorder():
    def __init__(self, video_output_dir: Path, logger):
        self.cap = cv2.VideoCapture(0)
        self.state = RecordingState.NOT_RECORDING
        self.video_writer = None
        self.video_dir = video_output_dir
        self.logger = logger
        self.video_logger = None
        self.last_detection_time = datetime.now()
        self.max_delta_history = deque(maxlen=50)
        self.contour_area_history = deque(maxlen=50)
        self.contour_area_history.extend([0]*50)

        self._clear_first_frames()
        _, self.latest_frame  = self.cap.read()
        self.motion_detector = MotionDetector(self.latest_frame)
        self.frame_event = threading.Event()
        self.frame_event.set()

    def get_latest_frame(self):
        return self.latest_frame

    def _clear_first_frames(self):
        for i in range(10):
            self.cap.read()
            time.sleep(0.5)

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                self.logger.error("Reading from frame failed!")
                time.sleep(1)
            self.latest_frame = frame
            self._process(frame)

    def _moving_avg_max_delta(self):
        return sum(self.max_delta_history) / len(self.max_delta_history)

    def _moving_avg_contour_area(self, last_n=None):
        if last_n is None:
            return sum(self.contour_area_history) / len(self.contour_area_history)
        return sum([self.contour_area_history[-i] for i in range(1, last_n+1)]) / last_n


    def _process(self, frame):
        t = datetime.now()
        results = self.motion_detector.detect(frame)
        self.frame_event.set()
        contour_area = results['Contour Analysis']['total_contour_area']
        self.max_delta_history.append(results['Raw Delta']['max_change'])
        self.contour_area_history.append(contour_area)

        match self.state:
            case RecordingState.NOT_RECORDING:
                if contour_area >= 700:
                    self._prepare_recording(t.strftime(utils.DATETIME_FORMAT))
                    self._record(frame, results, t)
                    self.last_detection_time = t
                else:
                    time.sleep(0.2)
            case RecordingState.RECORDING:
                if self._moving_avg_contour_area() < 300:
                    self.state = RecordingState.GRACE_PERIOD
                self._record(frame, results, t)
                self.last_detection_time = t
                return
            case RecordingState.GRACE_PERIOD:
                if self._moving_avg_contour_area() >= 300:
                    self.state = RecordingState.RECORDING
                    self._record(frame, results, t)
                    self.last_detection_time = t
                elif self._moving_avg_max_delta() < 3 or (t - self.last_detection_time).total_seconds() > 10:
                    self._stop_recording()
                else:
                    self._record(frame, results, t)


    def _prepare_recording(self, video_id):
        self.state = RecordingState.RECORDING
        self.video_logger = VideoLogger(video_id)
        self.video_writer = VideoWriter(str(self.video_dir / (video_id + '.mp4')))
        self.logger.info(f"Recording started for {video_id}")

    def _record(self, frame, motion_detection_results, t):
        self.video_writer.write(frame)
        log_entry = {
            'timestamp': t.strftime(utils.DATETIME_FORMAT_READABLE_SECOND),
            'motion_detection_results': motion_detection_results,
            'moving_avg_max_delta': self._moving_avg_max_delta(),
            'moving_avg_contour_area': self._moving_avg_contour_area(),
            'recording_state': self.state.value
        }
        self.video_logger.log(log_entry)

    def _stop_recording(self):
        self.video_writer.release()
        self.video_logger.close()
        self.state = RecordingState.NOT_RECORDING
        self.logger.info(f"Recording stopped")

    def is_recording(self):
        return self.state != RecordingState.NOT_RECORDING
    
    def cleanup(self):
        if self.is_recording():
            self._stop_recording()
        self.cap.release()
    
