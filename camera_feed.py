import cv2
import time
import threading

from video_utils import VideoWriter

class CameraFeed():
    def __init__(self, logger):
        self.logger = logger
        self.cap = cv2.VideoCapture(0)
        self.latest_frame = None
        self.is_running = False
        self.is_recording = False
        self.frame_lock = threading.Lock()
        self.frame_event = threading.Event()
        self.video_writer = None
        self._clear_first_frames()


    def _clear_first_frames(self):
        for _ in range(10):
            self.current_frame = self.cap.read()
            time.sleep(0.5)
    
    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_frames)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.is_running = False
        if self.is_recording:
            self.stop_recording()
        self.thread.join()

    def _capture_frames(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame
                    self.frame_event.set()
                if self.is_recording:
                    self.video_writer.write(frame)
            time.sleep(0.05)

    def start_recording(self, output_path):
        self.logger.info(f"Recording started for {output_path}")
        self.video_writer = VideoWriter(output_path)
        self.is_recording = True

    def stop_recording(self):
        self.logger.info(f"Recording stopped")
        self.is_recording = False
        self.video_writer.release()

    def stream_frame(self):
        while True:
            self.frame_event.wait()
            self.frame_event.clear()
            with self.frame_lock:
                yield self.latest_frame

    def get_frame(self):
        with self.frame_lock:
            return self.latest_frame
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
