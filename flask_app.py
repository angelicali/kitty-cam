import cv2
from ultralytics import YOLO
from flask import Flask, jsonify, request, send_from_directory, make_response
import flask
import atexit
import time
from datetime import datetime
import json
import threading
import os
from collections import Counter
import gc 

## Model
model = YOLO("finetuned_ncnn_model")

## Flask App
app = flask.Flask('xiaomao-cam', static_folder='static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 604800

# Camera
cap = cv2.VideoCapture(0)

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
logs = []

def cleanup():
    global video_labels, logs
    save_video_labels(video_labels)
    flush_logs(logs)
    cap.release()

atexit.register(cleanup)

DATETIME_FORMAT = '%Y%m%d%H%M%S'
DETECTION_BLACKLIST = {'bowl', 'potted plant', 'vase', 'surfboard', 'keyboard', 'bench'}

latest_frame = None
# latest_frame_backsub = None
recorded_frames = []
recording_start_time = None

def near(box1, box2, delta=4):
    for p in ['x1','y1','x2','y2']:
        if abs(box1[p] - box2[p]) > delta:
            return False
    return True


IMPOSSIBLE_LOC = {'x1':10, 'y1': 14, 'x2':142, 'y2':148}
BIRD_LOC = {'x1':523, 'y1':36, 'x2':546, 'y2':59}

def filter_results(results):
    objects = json.loads(results.to_json())
    filtered = []
    for obj in objects:
        if obj['name'] in DETECTION_BLACKLIST:
            continue

        if near(obj['box'], IMPOSSIBLE_LOC):
            continue

        if obj['name']=='bird' and near(obj['box'], BIRD_LOC, delta=1):
            continue
        
        filtered.append(obj)
    return filtered
        

def log_detected_activity(t, results, logs, detection_cnt):
    objects = json.loads(results.to_json())
    for obj in objects:
        detection_cnt[obj['name']] += 1

    timestr = t.strftime(DATETIME_FORMAT)
    logs.append((timestr,str(objects)))
    return True

def save_frames(frames, start_time, detection_cnt):
    if sum(detection_cnt.values()) <= 1:
        return

    timestr = start_time.strftime(DATETIME_FORMAT)
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_writer = cv2.VideoWriter('./static/' + timestr + '.mp4', fourcc, 20.0, (width, height))
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()

    with open(f'./logs/byvideo/{timestr}.json', 'w') as f:
        json.dump(detection_cnt, f)

def flush_logs(logs):
    if len(logs) == 0:
        return
    t0 = logs[0][0]
    t1 = logs[-1][0]
    filename = f"./logs/{t0}-{t1}.txt"
    with open(filename, 'w') as f:
        lines = [f"{t} | {objects}\n" for t, objects in logs]
        f.writelines(lines)


DAYTIME_GAP = 10 # seconds
NIGHT_GAP = 15 # seconds

def get_gap_tolerance(t):
    if 7 < t.hour < 20:
        return DAYTIME_GAP
    else:
        return NIGHT_GAP

def run_camera():
    global latest_frame, recorded_frames, logs
    last_detected = datetime.now()
    recording = False
    detection_cnt = Counter()
    while True:
        ret, frame = cap.read()
        t = datetime.now()
        if not ret:
            print("reading from camera failed")
            time.sleep(1)
            continue
        
        results = model(frame, verbose=False)[0]
        latest_frame = results.plot()
        # latest_frame_backsub = back_sub.apply(frame)

        # Check current frame
        if log_detected_activity(t, results, logs, detection_cnt):
            if not recording:
                recording = True
                recording_start_time = t
            last_detected = t

        # If it's been a while since last detected anything, stop recording
        since_last_detection = t - last_detected
        if recording and since_last_detection.total_seconds() > get_gap_tolerance(t): 
            recording = False
            save_frames(recorded_frames, recording_start_time, detection_cnt)
            recorded_frames = []
            detection_cnt = Counter()
            gc.collect()

        # If recording (i.e. either current frame detected, or last detection was only a little bit ago)
        if recording:
            recorded_frames.append(frame)


        if len(logs)>=500:
            flush_logs(logs)
            logs = []
            gc.collect()
            

def get_video_feed(cap):
    while True:
        frame = latest_frame
        if frame is not None:
            ret, buf = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
				+ buf.tobytes() + b'\r\n')


@app.route('/video_feed')
def video_feed():
	return flask.Response(get_video_feed(cap), mimetype='multipart/x-mixed-replace;boundary=frame')

# @app.route('/video_feed_backsub')


@app.route('/')
def index():
	return flask.render_template('index.html')

# TODO: retrieve video labels too
def get_videos(max_videos=None):
    video_files = os.listdir('./static')
    video_files.sort(reverse=True)
    time_and_video = []
    video_files = video_files if max_videos==None else video_files[:max_videos]
    for v in video_files:
        timestr = v.split('.')[0]
        dt = datetime.strptime(timestr, DATETIME_FORMAT)
        time_and_video.append((dt.strftime('%Y-%m-%d %H:%M'), timestr))
    return time_and_video

@app.route('/activities')
def activities():
    time_and_videoid = get_videos(max_videos=35)
    time_and_videoid = [(t,v) for t,v in time_and_videoid if video_labels.get(v, '') not in LABELS_TO_HIDE][:20]
    decoded_video_labels = {v:LABEL_CODES_DISPLAY[video_labels[v]] for _, v in time_and_videoid if v in video_labels} 
    return flask.render_template('videos.html', video_files=time_and_videoid, video_labels=decoded_video_labels)

@app.route('/video/<path:filename>')
def serve_video(filename):
    response = make_response(send_from_directory('./static/', filename))
    response.headers['Cache-Control'] = 'public, max-age=604800, immutable'
    return response

# =====  Admin routes  =====

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
