import ffmpeg
import logging
import json
import threading

import utils

class VideoWriter():
    """ For writing a single video. """
    def __init__(self, video_id):
        output_path = str(utils.VIDEO_DIR / (video_id + '.mp4'))

        self.process = (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='640X480', r=20.0)
                .output(output_path, pix_fmt='yuv420p', vcodec='libx264')
                .overwrite_output()
                .run_async(pipe_stdin=True)
                )
        self.is_active = True
        
    def write(self, frame):
        if self.is_active:
            self.process.stdin.write(frame.tobytes())

    def release(self):
        self.is_active = False
        self.process.stdin.close()
        self.process.wait()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

class VideoLogger():
    def __init__(self, video_id):
        self.logger = logging.getLogger(f"detection_logger")
        self.logger.setLevel(logging.INFO)
        
        self.file_handler = logging.FileHandler(utils.VIDEO_LOG_DIR / f"{video_id}.jsonl", mode='a')
        self.logger.addHandler(self.file_handler)
    
    def log(self, data):
        self.logger.info(json.dumps(data))

    def close(self):
        self.logger.removeHandler(self.file_handler)
        if self.file_handler is not None:
            self.file_handler.close()

class VideoLoggerHandler():
    def __init__(self):
        self.video_logger = None
        self.lock = threading.Lock()

    def create_logger(self, video_id):
        with self.lock:
            self.video_logger = VideoLogger(video_id)
    
    def log(self, data):
        with self.lock:
            if self.video_logger is not None:
                self.video_logger.log(data)
    
    def close_logger(self):
        with self.lock:
            self.video_logger.close()
            self.video_logger = None