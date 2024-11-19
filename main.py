import multiprocessing.shared_memory
from flask_app import app as flask_app
import atexit
from datetime import datetime
import logging
import multiprocessing

from camera_feed import CameraFeed
from detection_manager import DetectionManager

if __name__ == '__main__':
    today = str(datetime.now().date())

    # Logging
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(f'logs/{today}.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


    # CameraFeed owns VideoWriter
    camera_feed = CameraFeed(logger)
    # DetectionManager owns MotionDetector, ObjectDetector, and VideoLoggerHandler
    detection_manager = DetectionManager(camera_feed)

    def cleanup():
        detection_manager.stop()
        camera_feed.stop()

    atexit.register(cleanup)

    camera_feed.start()
    detection_manager.start()
    
    flask_app.camera_feed = camera_feed
    flask_app.logger.addHandler(file_handler)
    flask_app.run(host='0.0.0.0', port=5000)
