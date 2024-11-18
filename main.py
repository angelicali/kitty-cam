from flask_app import app as flask_app
import atexit
from datetime import datetime
import logging

from camera_feed import CameraFeed
from detection_manager import DetectionManager

if __name__ == '__main__':
    # Logging
    today = str(datetime.now().date())
    logger = logging.getLogger(__name__)
    log_filename = f"logs/{today}.log"
    logging.basicConfig(filename=log_filename, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
    flask_app.logger = logger
    flask_app.run(host='0.0.0.0', port=5000)
