import cv2
from datetime import datetime
from flask import Flask, Response, request, make_response, send_from_directory, stream_with_context
from flask_cors import CORS
import logging 
import os
import threading
import utils
import base64
import json

HOME_IP = os.getenv("HOME_IP")

# Flask app
app = Flask("kitty-cam", static_folder="static")
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 604800
CORS(app) # TODO: is this correct?

### API routes  
def _get_livestream():
    for frame in app.camera_feed.stream_frame():

        _, buf = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image.jpeg\r\n\r\n'
               + buf.tobytes() + b'\r\n')


def _get_livestreamr():
    for frame in app.camera_feed.stream_frame():
        _, buf = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buf).decode('utf-8')
        data = json.dumps({
            "frame": frame_base64,
            "is_recording": app.camera_feed.get_is_recording()
        })
        
        yield f"data: {data}\n\n"

@app.route('/livestream')
def livestream():
    return Response(_get_livestream(), mimetype='multipart/x-mixed-replace;boundary=frame')

@app.route('/livestreamr')
def livestreamr():
    return Response(stream_with_context(_get_livestreamr()), mimetype='text/event-stream')

@app.route('/past-visits')
def past_visists():
    n_videos = request.args.get('n', 200)
    prefix = request.args.get('prefix', None)
    return utils.get_video_list(skip_latest=app.camera_feed.get_is_recording(), max_videos=n_videos, return_id=True, prefix=prefix)

def is_user_admin(request):
    user_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    authorized = user_ip.startswith('192.168.') or user_ip == HOME_IP
    if not authorized:
        app.logger.debug(f"unauthorized ip: {user_ip}")
    return authorized

@app.route('/video/<path:video_id>', methods=['GET', 'DELETE'])
def video_request(video_id):
    if request.method == 'GET':
        response = make_response(send_from_directory(str(utils.VIDEO_DIR), f"{video_id}.mp4"))
        response.headers['Cache-Control'] = 'public, max-age=604800, must-revalidate'
        return response
    elif request.method == 'DELETE':
        if not is_user_admin(request):
            return {"error": f"Unauthorized"}, 403
        utils.delete_video_by_id(video_id)
        return f"deleted {video_id}"

@app.route('/favorites')
def get_favorites():
    return utils.get_favorites()

@app.route('/favorite/<path:video_id>', methods=['POST', 'DELETE'])
def set_favorite(video_id):
    if not is_user_admin(request):
        return {"error": f"Unauthorized"}, 403
    if request.method == 'POST':
        utils.set_favorite(video_id)
        return f"marked {video_id} as favorite"
    elif request.method == 'DELETE':
        utils.set_favorite(video_id, delete=True)
        return f"unmarked {video_id} as favorite"

@app.route('/merge', methods=['POST'])
def merge_videos():
    if not is_user_admin(request):
        return {"error": f"Unauthorized"}, 403
    utils.merge(request.form.getlist('video_to_merge'))
    return f"merged videos"

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

