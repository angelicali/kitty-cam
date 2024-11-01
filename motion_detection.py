import cv2

class MotionDetector():
    def __init__(self, initial_frame, blur_size=21, threshold=25, min_area=500):
        self.prev_frame_blurred =  MotionDetector.blur(initial_frame)
        self.blur_size = blur_size
        self.threshold = threshold
        self.min_area = min_area

    def _blur(self, frame):
        # TODO: is the original frame modified?
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
    
    def detect(self, frame):
        blurred_frame = self._blur(frame)
        frame_delta = cv2.absdiff(self.prev_frame_blurred, blurred_frame)
        self.prev_frame_blurred = blurred_frame
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant_motion = False
        motion_regions = []

        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area > self.min_area:
                significant_motion = True
                (x, y, w, h) = cv2.boundingRect(contour)
                motion_regions.append({'x': x, 'y': y, 'width': w, 'height': h, 'area': contour_area})
        
        return {
            "motion_detected": significant_motion,
            "regions": motion_regions
        }
    
    def set_blur_size(self, blur_size):
        self.blur_size = blur_size

    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_min_area(self, min_area):
        self.min_area = min_area
    
    def get_configs(self):
        return {"blur_size": self.blur_size, "threshold": self.threshold, "min_area": self.min_area}
    