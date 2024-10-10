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
DATETIME_FORMAT_READABLE = '%Y/%m/%d %H:%M'

latest_frame = None
# latest_frame_backsub = None
recorded_frames = []
recording_start_time = None
last_activity_time = datetime.strptime(sorted(os.listdir('./static'), reverse=True)[0].split('.')[0], DATETIME_FORMAT)

def log_detected_activity(t, objects, logs, detection_cnt, threshold=0.4):
    if len(objects)==0:
        return []

    filtered_objects = []
    for obj in objects:
        if obj['name'] == "tabby":
            continue
        if obj['confidence'] >= threshold:
            detection_cnt[obj['name']] += 1
            filtered_objects.append(obj)

    timestr = t.strftime(DATETIME_FORMAT)
    logs.append((timestr,str(objects)))

    return filtered_objects

def write_video(filename, frames):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()


def save_frames(frames, start_time, detection_cnt):
    if sum(detection_cnt.values()) <= 2:
        return

    global last_activity_time
    timestr = start_time.strftime(DATETIME_FORMAT)
    write_video(f"./static/{timestr}.mp4", frames)
    with open(f'./logs/byvideo/{timestr}.json', 'w') as f:
        json.dump(detection_cnt, f)
    last_activity_time = start_time

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
    if 7 <= t.hour <= 20:
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
        
        # 0.88 MB per frame
        # frame_memory = frame.nbytes / (1024*1024)
        # print(f"Frame size: {frame_memory: .2f} MB")

        results = model(frame, verbose=False)[0]
        objects = json.loads(results.to_json())
        filtered_objects = log_detected_activity(t, objects, logs, detection_cnt, threshold=0.35)

        if len(filtered_objects)!=0:
            latest_frame = results.plot()
            if not recording:
                gc.collect()
                recording = True
                recording_start_time = t
            last_detected = t
            if len(logs)>=100:
                flush_logs(logs)
                logs = []
                gc.collect()
        else:
            latest_frame = frame
            if not recording:
                time.sleep(0.75)

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
            if len(recorded_frames) >= 1000:
                save_frames(recorded_frames, recording_start_time, detection_cnt)
                recorded_frames = []
                detection_cnt = Counter()
                recording_start_time = datetime.now()
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
	return flask.render_template('index.html', last_activity_time=last_activity_time.strftime(DATETIME_FORMAT_READABLE))

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



@app.route('/video/<path:filename>')
def serve_video(filename):
    replay = request.args.get('replay', default = False, type=bool)
    if replay:
        annotated_filename = 'annotated_' + filename
        if not os.path.exists('./static/' + annotated_filename):
            annotate_video(filename)
        filename = annotated_filename
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
