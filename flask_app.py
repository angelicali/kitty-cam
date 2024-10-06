import cv2
from ultralytics import YOLO
import flask
import atexit
import time
from datetime import datetime
import json
import threading
import os

## Model
model = YOLO("yolov5nu_ncnn_model")

## Flask App
app = flask.Flask('xiaomao-cam', static_folder='static')

# Camera
cap = cv2.VideoCapture(0)

def cleanup():
    cap.release()

atexit.register(cleanup)

DATETIME_FORMAT = '%Y%m%d%H%M%S'
DETECTION_BLACKLIST = {'bowl', 'potted plant', 'vase', 'surfboard', 'keyboard', 'bench'}

latest_frame = None
recorded_frames = []
recording_start_time = None

def log_detected_activity(t, results, logs):
    objects = json.loads(results.to_json())
    object_names = set([obj['name'] for obj in objects])
    object_names -= DETECTION_BLACKLIST
    if len(object_names) == 0:
        return False
    timestr = t.strftime(DATETIME_FORMAT)
    logs.append((timestr,','.join(sorted(object_names))))
    return True

def save_frames(frames, start_time):
    if len(frames) <= 1:
        return

    timestr = start_time.strftime(DATETIME_FORMAT)
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_writer = cv2.VideoWriter('./static/' + timestr + '.mp4', fourcc, 20.0, (width, height))
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()

def flush_logs(logs):
    t0 = logs[0][0]
    t1 = logs[-1][0]
    filename = f"./logs/{t0}-{t1}.txt"
    with open(filename, 'w') as f:
        lines = [f"{t} | {objects}\n" for t, objects in logs]
        f.writelines(lines)


DAYTIME_GAP = 5 # seconds
NIGHT_GAP = 15 # seconds

def get_gap_tolerance(t):
    if 7 < t.hour < 20:
        return DAYTIME_GAP
    else:
        return NIGHT_GAP

def run_camera():
    global latest_frame, recorded_frames
    logs = []
    last_detected = datetime.now()
    recording = False
    while True:
        ret, frame = cap.read()
        t = datetime.now()
        if not ret:
            print("reading from camera failed")
            time.sleep(1)
            continue
        
        results = model(frame, verbose=False)[0]
        latest_frame = results.plot()

        # Check current frame
        if log_detected_activity(t, results, logs):
            if not recording:
                recording = True
                recording_start_time = t
            last_detected = t

        # If it's been a while since last detected anything, stop recording
        since_last_detection = t - last_detected
        if recording and since_last_detection.total_seconds() > get_gap_tolerance(t): 
            recording = False
            save_frames(recorded_frames, recording_start_time)
            recorded_frames = []

        # If recording (i.e. either current frame detected, or last detection was only a little bit ago)
        if recording:
            recorded_frames.append(frame)


        if len(logs)>=5000:
            flush_logs(logs)
            logs = []
            

def get_video_feed(cap):
    global latest_frame
    while True:
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
	return flask.render_template('index.html')

@app.route('/activities')
def activities():
    video_files = os.listdir('./static')
    video_files.sort(reverse=True)
    time_and_video = []
    video_files = video_files[:20]
    for v in video_files:
        timestr = v.split('.')[0]
        dt = datetime.strptime(timestr, DATETIME_FORMAT)
        time_and_video.append((dt.strftime('%Y-%m-%d %H:%M'), v))

    return flask.render_template('videos.html', video_files=time_and_video)

@app.route('/static/<path:filename>')
def serve_video(filename):
    return flask.send_from_directory('./static/', filename)

if __name__ == '__main__':
    camera_thread = threading.Thread(target=run_camera)
    camera_thread.daemon = True
    camera_thread.start()
    
    app.run(host='0.0.0.0', port=5000)
