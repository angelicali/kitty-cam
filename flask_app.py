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


latest_frame = None
detected_frames = []
detection_start_time = None
last_detected = datetime.now()

def log_detected_activity(t, results, logs):
    objects = json.loads(results.to_json())
    object_names = [obj['name'] for obj in objects]
    if 'bowl' in object_names:
        object_names.remove('bowl')
    if len(object_names) == 0:
        return False
    timestr = t.strftime('%Y%m%d%H%M%S')
    logs.append((timestr,','.join(object_names)))
    return True

def save_frames(frames, detection_start_time):
    timestr = detection_start_time.strftime('%Y%m%d%H%M%S')
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

def run_camera():
    global latest_frame, detected_frames
    logs = []
    while True:
        ret, frame = cap.read()
        t = datetime.now()
        if not ret:
            print("reading from camera failed")
            time.sleep(1)
            continue
        
        results = model(frame, verbose=False)[0]
        latest_frame = results.plot()
        
        if log_detected_activity(t, results, logs):
            if len(detected_frames) == 0:
                detection_start_time = datetime.now()
            detected_frames.append(frame)
            last_detected = datetime.now()
        elif len(detected_frames)>0:
            since_last_detection = datetime.now() - last_detected
            if since_last_detection.total_seconds() > 10:
                save_frames(detected_frames, detection_start_time)
                detected_frames = []

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
    return flask.render_template('videos.html', video_files=video_files)

@app.route('/static/<path:filename>')
def serve_video(filename):
    return flask.send_from_directory('./static/', filename)

if __name__ == '__main__':
    camera_thread = threading.Thread(target=run_camera)
    camera_thread.daemon = True
    camera_thread.start()
    
    app.run(host='0.0.0.0', port=5000)
