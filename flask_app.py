import logging
import cv2
from ultralytics import YOLO
from flask import Flask, jsonify, request, send_from_directory, make_response
import flask
from flask_cors import CORS
import atexit
import time
from datetime import datetime
import json
import threading
import os
from collections import Counter
import gc 
import ffmpeg
from pathlib import Path

HOME_IP = os.getenv("HOME_IP")
print("home ip is:", HOME_IP)
## Model
model = YOLO("finetuned_ncnn_model")

## Flask App
app = flask.Flask('xiaomao-cam', static_folder='static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 604800
CORS(app)

# Camera
cap = cv2.VideoCapture(0)

# Logging
DATETIME_FORMAT = '%Y%m%d%H%M%S'
DATETIME_FORMAT_READABLE = '%Y/%m/%d %H:%M'
DATETIME_FORMAT_READABLE_SECOND = '%Y/%m/%d %H:%M:%S'

today = str(datetime.now().date())
log_filename = f"./logs/{today}.log"
logger = logging.getLogger(__name__)
logging.basicConfig(filename=log_filename, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

recording = False

def get_videos(max_videos=None):     
    video_files = os.listdir('./static')
    video_files.sort(reverse=True)          
    time_and_video = []               
    if recording:          
        video_files = video_files[1:]
    if max_videos is not None:           
        video_files = video_files[:max_videos]
    for v in video_files:                         
        timestr = v.split('.')[0]
        dt = datetime.strptime(timestr, DATETIME_FORMAT)
        time_and_video.append((dt.strftime('%Y-%m-%d %H:%M'), timestr))
    return time_and_video # readable timestamp (as header) and video id

def get_video_logs():
    logs = {}
    videos = get_videos()    
    for _, video_id in videos:    
        if video_id < "20241026213043":    
            break    
        logfile = Path(f"logs/byvideo/{video_id}.json")    
        if not logfile.exists():    
            continue    
        with logfile.open('r') as f:
            logs[video_id] = json.load(f)
    return logs


video_logs = get_video_logs()


# Video labels
def get_video_labels():
    with open('video_labels.json', 'r') as f:
        return json.load(f)

def save_video_labels(video_labels):
    with open('video_labels.json', 'w') as f:
        json.dump(video_labels, f)

video_labels = get_video_labels()
LABEL_CODES_DISPLAY = {'xiaomao': "Xiao mao", "siama": "Siama", "possum": "Possum", "racoon": "Racoon", 'feeder':"Feeder", 'tabby': 'Tabby'}
LABEL_CODES_SELECT  = LABEL_CODES_DISPLAY.copy()
LABEL_CODES_SELECT.update({'fp': "False Positive", 'feeder':"Feeder", "person":"Person", "dogwalker": "Dog Walker"})
LABELS_TO_HIDE = {"fp", "person", "dogwalker"}

# Save video detection logs
def save_video_detections(video_id, detections):
    with open(f'logs/byvideo/{video_id}.json', 'w') as f:
        json.dump(detections, f)
    
    global video_logs
    video_logs[video_id] = detections

def cleanup():
    global video_labels
    save_video_labels(video_labels)
    cap.release()

atexit.register(cleanup)


# Frame livestreaming 
frame_event = threading.Event()
latest_frame = None
latest_detection_time = datetime.strptime(sorted(os.listdir('./static'), reverse=True)[0].split('.')[0], DATETIME_FORMAT)


def update_frame(new_frame):
    global latest_frame
    latest_frame = new_frame
    frame_event.set()

def pass_threshold(obj):
    return obj['confidence'] >= 0.1

class VideoWriter():
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


def run_camera():
    global latest_frame, recorded_frames, recording
    last_detected = datetime.now()
    video_writer = None
    video_detections = []
    video_id = None

    def _start_or_keep_recording(t, frame):
        if frame is None:
            return

        nonlocal video_writer, video_id, video_detections
        if video_writer is None:
            logger.info('Recording started')
            video_id = t.strftime(DATETIME_FORMAT)
            video_writer = VideoWriter(f'./static/{video_id}.mp4')
            video_detections = [video_detections[-1]] # it must  be non-empty, since detection is triggered

        video_writer.write(frame)

    def _stop_recording():
        global latest_detection_time, recording
        nonlocal video_writer, video_detections, video_id
        recording = False
        video_writer.release()
        video_writer = None
        latest_detection_time = datetime.now()
        logger.info('Recording stopped')
        save_video_detections(video_id, video_detections)
        video_detections = []

    while True:
        ret, frame = cap.read()
        t = datetime.now()
        if not ret:
            logger.warning("reading from camera failed")
            time.sleep(1)
            continue
        
        if frame.nbytes < 900000:
            logger.warning(f"frame.nbytes: {frame.nbytes}")

        # 0.88 MB per frame
        # frame_memory = frame.nbytes / (1024*1024)
        # print(f"Frame size: {frame_memory: .2f} MB")

        results = model(frame, verbose=False)[0]
        objects = json.loads(results.to_json())
        
        # Log objects if any, and update livestream frame
        if len(objects) != 0:
            logger.info(str(objects))
            update_frame(results.plot())
            video_detections.append((t.strftime(DATETIME_FORMAT_READABLE_SECOND), objects))
        else:
            update_frame(frame)
        
        # Recording logic, depending on whether detected objects pass threshold
        filtered_objects = [obj['name'] for obj in objects if pass_threshold(obj)]
        
        # If detected anything this frame, record frame and continue
        if len(filtered_objects)>0:
            recording = True
            last_detected = t
            _start_or_keep_recording(t, frame)
            continue

        # If didn't detect anything this frame:
        # Case 1: wasn't recording: sleep and continue
        if not recording:
            time.sleep(0.25)
            continue 
        # Case 2: within gap tolerance: keep recording
        if (t - last_detected).total_seconds() <= 15:
            _start_or_keep_recording(t, frame)
        # Case 3: past gap tolerance: stop recording
        else:
            logger.debug(f"current time: {t}; last_detected time: {last_detected}")
            _stop_recording()


def get_video_feed(cap):
    while True:
        frame_event.wait()
        frame_event.clear()

        if latest_frame is not None:
            ret, buf = cv2.imencode('.jpg', latest_frame)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
				+ buf.tobytes() + b'\r\n')


@app.route('/video_feed')
def video_feed():
	return flask.Response(get_video_feed(cap), mimetype='multipart/x-mixed-replace;boundary=frame')


@app.route('/')
def index():
	return flask.render_template('index.html', latest_detection_time=latest_detection_time.strftime(DATETIME_FORMAT_READABLE))

@app.route('/past-visits')
def past_visists_metadata():
    return get_videos()

@app.route('/activities')
def activities():
    time_and_videoid = get_videos(max_videos=35)
    time_and_videoid = [(t,v) for t,v in time_and_videoid if video_labels.get(v, '') not in LABELS_TO_HIDE][:20]
    decoded_video_labels = {v:LABEL_CODES_DISPLAY[video_labels[v]] for _, v in time_and_videoid if v in video_labels} 
    return flask.render_template('videos.html', video_files=time_and_videoid, video_labels=decoded_video_labels)


def annotate_video(filename):
    print(f"annotating {filename}")
    v = cv2.VideoCapture('./static/' + filename)
    started = False
    attempts_left = 10
    frames = []
    while v.isOpened():
        ret, frame = v.read()
        if not ret:
            if started or attempts_left <= 0:
                break
            else:
                time.sleep(1)
                attempts_left -= 1
                continue
        started = True        
        results = model(frame)[0]
        frames.append(results.plot())
    write_video("annotated_" + filename, frames)



@app.route('/video/<path:filename>', methods=['GET', 'DELETE'])
def serve_video(filename):
    if request.method == 'GET':
        replay = request.args.get('replay', default = False, type=bool)
        if replay:
            annotated_filename = 'annotated_' + filename
            if not os.path.exists('./static/' + annotated_filename):
                annotate_video(filename)
            filename = annotated_filename
        response = make_response(send_from_directory('./static/', filename))
        response.headers['Cache-Control'] = 'public, max-age=604800, immutable'
        return response
    elif request.method == 'DELETE':
        user_ip = request.remote_addr
        if user_ip != HOME_IP:
            return {"error": "Unauthorized"}, 403

        filename = Path(filename)
        video_id = filename.stem
        video_dir = Path('static/')
        log_dir = Path('logs/byvideo/')
        video_path = video_dir / filename
        video_log_path = log_dir / f"{video_id}.json"
        if not video_path.exists():
            return {"error": f"file {filename} not found"}, 404
        
        trash_bin = Path('trash-bin')
        video_path.rename(trash_bin / filename)
        if video_log_path.exists():
            video_log_path.rename(trash_bin / video_log_path.stem)
            global video_logs
            del video_logs[video_id]
        logger.info(f"deleting: moving {video_path} to {trashed_path}")
        return f"Moved {filename} to trash bin"

@app.route('/undo-delete/<path:filename>', methods=['POST'])
def undo_delete(filename):
    user_ip = request.remote_addr
    if user_ip != HOME_IP:
        return {"error": "Unauthorized"}, 403

    filename = Path(filename)
    video_id = filename.stem
    video_dir = Path('static/')
    log_dir = Path('logs/byvideo/')
    trash_bin = Path('trash-bin')
    target_video_path = video_dir / filename
    target_video_log_path = log_dir / f"{video_id}.json"
    
    video_path = trash_bin / filename
    video_log_path = trash_bin / f"{video_id}.json"
    if video_path.exists():
        video_path.rename(target_video_path)
        if video_log_path.exists():
            video_log_path.rename(target_video_log_path)
            load_video_log(video_id)
        logger.info(f"undoing delete: moving {video_path} to {target_path}")
    else:
        return {"error": f"file {filename} not found"}, 404

@app.route('/video-log/<path:video_id>')
def video_log(video_id):
    if video_id not in video_logs:
        return {"error": f"log for video {video_id} not found"}, 404
    else:
        return video_logs[video_id]

def load_video_log(video_id):
    global video_logs
    logfile = f"logs/byvideo/{video_id}.json"
    with logfile.open('r') as f:
        video_logs[video_id] = json.load(f)

@app.route('/locations/all')
def locations_by_class():
    results = {}
    for video_id, logs in video_logs.items():
        for timestamp, detections in logs:
            for d in detections:
                if d['name'] not in results:
                    results[d['name']] = []
                results[d['name']].append({
                    'box': d['box'],
                    'confidence': d['confidence']
                    })
    return results

@app.route('/locations/<path:object_class>')
def locations(object_class):
    results = []
    for video_id, logs in video_logs.items():
        for timestamp, detections in logs:
            for d in detections:
                if d['name'] == object_class:
                    results.append(d["box"])
    return results

# =====  Admin routes  =====

@app.route('/logs')
def view_logs():
    with open(log_filename, 'r') as f:
        log_content = f.read()
    return flask.Response(log_content, mimetype='text/plain')

@app.route('/admin')
def admin():
    time_and_videoid = get_videos()
    page = request.args.get('page',1, type=int)
    per_page = 20

    start_idx = (page-1) * per_page
    end_idx = start_idx + per_page

    total_pages = (len(time_and_videoid) - 1) // per_page + 1

    # TODO: display also detected objects per video, for videos for this page
    return flask.render_template("admin.html", video_files=time_and_videoid[start_idx:end_idx], page=page, total_pages=total_pages, video_labels=video_labels, label_codes=LABEL_CODES_SELECT)

@app.route('/delete_videos', methods=['POST'])
def delete_videos():
    response = {"removed": [], "error": []}
    video_ids = request.form.getlist('checked_video_ids')
    for videoid in video_ids:
        try:
            filename = f'./static/{videoid}.mp4'
            os.remove(filename)
            response["removed"].append(filename)
        except OSError as error:
            response["error"].append(error)
    return jsonify(response)
    # return redirect('/admin')

@app.route('/save_labels', methods=['POST'])
def save_labels():
    global video_labels
    response = {"updates":[]}
    for key in request.form:
        if key.startswith('label_'):
            videoid = key.replace('label_', '')
            selected_label = request.form[key]
            video_labels[videoid] = selected_label
            response['updates'].append((videoid, selected_label))
    response['updated_video_labels'] = video_labels
    return jsonify(response)
    # return redirect('/admin')

if __name__ == '__main__':
    camera_thread = threading.Thread(target=run_camera)
    camera_thread.daemon = True
    camera_thread.start()
    
    app.run(host='0.0.0.0', port=5000)
