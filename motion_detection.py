import cv2
import numpy as np

class MotionDetector():
    def __init__(self, initial_frame, blur_size=21, threshold=30, min_area=500):
        self.blur_size = blur_size
        self.threshold = threshold
        self.min_area = min_area
        self.prev_frame_blurred = self._blur(initial_frame)

    def _blur(self, frame):
        frame = frame.copy()
        # TODO: is the original frame modified?
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
    
    def detect(self, frame):
        blurred_frame = self._blur(frame)
        raw_delta = cv2.absdiff(self.prev_frame_blurred, blurred_frame)
        self.prev_frame_blurred = blurred_frame

        threshold_delta = cv2.threshold(raw_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        dilated_delta = cv2.dilate(threshold_delta, None, iterations=2)
        contours, _ = cv2.findContours(dilated_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        metics = {
            'Raw Delta': {
                'mean_change': np.mean(raw_delta),
                'max_change': np.max(raw_delta),
                'percent_changed': (raw_delta > self.threshold).mean() * 100
            },
            'Threshold Delta': {
                'mean_change': np.mean(threshold_delta),
                'max_change': np.max(threshold_delta),
                'percent_changed': (threshold_delta > 0).mean() * 100
            },
            'Dilated Delta': {
                'mean_change': np.mean(dilated_delta),
                'max_change': np.max(dilated_delta),
                'percent_changed': (dilated_delta > 0).mean() * 100
            },
            'Contour Analysis': {
                'total_contour_area': sum(cv2.contourArea(c) for c in contours),
                'contour_count': len(contours)
            }
        },

        metics["motion_detected"] = False
        metics['Motion Regions'] = []

        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area > self.min_area:
                metics["motion_detected"] = True
            (x, y, w, h) = cv2.boundingRect(contour)
            metics['Motion Regions'].append({'x': x, 'y': y, 'width': w, 'height': h, 'area': contour_area})
        
        return metics

    def set_blur_size(self, blur_size):
        self.blur_size = blur_size

    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_min_area(self, min_area):
        self.min_area = min_area
    
    def get_configs(self):
        return {"blur_size": self.blur_size, "threshold": self.threshold, "min_area": self.min_area}
    
