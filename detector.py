import cv2
from ultralytics import YOLO
import time
from datetime import datetime
import json

model = YOLO("yolov5nu_ncnn")
OBJECTS_OF_INTEREST = {"cat", "dog", "person"}

def detect(frame):
    result = model(frame)[0]
    obj_and_probs = []
    for i in range(5):
        obj = result.names[result.top5[i]]
        if obj not in OBJECTS_OF_INTEREST:
            continue
        prob = result.top5conf[i]
        obj_and_probs.append((obj, prob))
    return obj_and_probs

def write(t, frame, objects):
    filepath = './data/' + t.strftime('%Y%m%d%H%M%S')

    # write the image
    cv2.imwrite(filepath + '.jpg', frame)

    # write the json
    with open(filepath + '.json', 'w') as f:
        json.dump(objects, f)

def main():
    cam = cv2.VideoCapture(0)
    last_detection = 0
    try:
        while True :
            t = datetime.now()
            ret, frame = cam.read()
            if not ret:
                print("read frame failed!")
                time.sleep(10)
                continue
            
            objects = detect(frame)
            if len(objects) != 0:
                write(t, frame, objects)
                time.sleep(60) # if detected, sleep for a minute
            else:
                time.sleep(60*3) # if not detected, sleep for 3 minute
    except Exception as e:
        pass 
    
    cam.release()