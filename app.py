import atexit
import cv2
from datetime import datetime
from flask import Flask, Response, request, make_response, send_from_directory
from flask_cors import CORS
import logging 
import os
import threading
from recording import CameraRecorder
import utils

HOME_IP = os.getenv("HOME_IP")

# Flask app
app = Flask("kitty-cam", static_folder="static")
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 604800
CORS(app) # TODO: is this correct?


# Logging
today = str(datetime.now().date())
logger = logging.getLogger(__name__)
log_filename = f"logs/{today}.log"
logging.basicConfig(filename=log_filename, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


recorder = CameraRecorder(utils.VIDEO_DIR, logger)
atexit.register(recorder.cleanup)

### API routes  
def _get_livestream():
    while True:
        recorder.frame_event.wait()
        recorder.frame_event.clear()

        _, buf = cv2.imencode('.jpg', recorder.latest_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image.jpeg\r\n\r\n'
               + buf.tobytes() + b'\r\n')

@app.route('/livestream')
def livestream():
    return Response(_get_livestream(), mimetype='multipart/x-mixed-replace;boundary=frame')

@app.route('/past-visits')
def past_visists():
    return utils.get_video_list(skip_latest=recorder.is_recording(), max_videos=50, return_id=True)

def is_user_admin(request):
    user_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    authorized = user_ip.startswith('192.168.') or user_ip == HOME_IP
    if not authorized:
        logger.debug(f"unauthorized ip: {user_ip}")
    return authorized

@app.route('/video/<path:video_id>', methods=['GET', 'DELETE'])
def video_request(video_id):
    if request.method == 'GET':
        response = make_response(send_from_directory(str(utils.VIDEO_DIR), f"{video_id}.mp4"))
        response.headers['Cache-Control'] = 'public, max-age=604800, immutable'
        return response
    elif request.method == 'DELETE':
        if not is_user_admin(request):
            return {"error": f"Unauthorized"}, 403
        utils.delete_video_by_id(video_id)
        return f"deleted {video_id}"

@app.route('/video-log/<path:video_id>')
def video_log(video_id):
    return utils.get_video_log(video_id)

@app.route('/locations')
@app.route('/locations/all')
def locations():
    return utils.get_location_analytics()

@app.route('/active-hour')
def active_hour():
    return utils.get_active_hour_analytics()

@app.route('/logs')
def logs():
    with open(log_filename, 'r') as f:
        log_content = f.read()
    return Response(log_content, mimetype='text/plain')

if __name__ == '__main__':
    camera_thread = threading.Thread(target=recorder.run)
    camera_thread.daemon = True
    camera_thread.start()
    
    app.run(host='0.0.0.0', port=5000)
