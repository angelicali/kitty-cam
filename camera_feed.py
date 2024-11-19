import cv2
import time
import threading
import multiprocessing

from video_utils import VideoWriter

class CameraFeed():
    def __init__(self, logger, camera_source=0):
        self.logger = logger
        self.cap = cv2.VideoCapture(camera_source)
        self.latest_frame = None
        self.is_running = False
        self.is_recording = multiprocessing.Value('i', 0)
        self.frame_lock = threading.Lock()
        self.frame_event = threading.Event()
        self.video_writer = None
        self._clear_first_frames()

        # for object detection only, to pass frame to its process
        self.frame_queue = multiprocessing.Queue()
    
    def get_is_recording(self):
        return self.is_recording.value == 1

    def _clear_first_frames(self):
        for _ in range(3):
            ret, self.latest_frame = self.cap.read()
            time.sleep(0.05)

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_frames)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        print("stopping CameraFeed...")
        self.is_running = False
        if self.get_is_recording():
            self.stop_recording()
        self.thread.join()
        print("CameraFeed thread joined")
        print("Closing frame queue...")
        while not self.frame_queue.empty():
            self.frame_queue.get()
        self.frame_queue.close()
        self.frame_queue.join_thread()
        print("frame queue closed")

    def _capture_frames(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame
                    self.frame_event.set()
                if self.get_is_recording():
                    self.video_writer.write(frame)
                if self.frame_queue.empty():
                    self.frame_queue.put(frame)
            time.sleep(0.05)

    def start_recording(self, output_path):
        self.logger.info(f"Recording started for {output_path}")
        self.video_writer = VideoWriter(output_path)
        self.is_recording.value = 1

    def stop_recording(self):
        self.is_recording.value = 0
        self.video_writer.release()
        self.logger.info(f"Recording stopped")

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
