import utils
import random
from datetime import datetime

OBJECT_TYPES = ["cat", "raccoon", "possum"]

def generate_location_analytics(logs):
    locations = {o: [] for o in OBJECT_TYPES}
    for video_log in logs.values():
        for _, detection in video_log:
            for d in detection:
                if d['name'] not in locations:
                    continue
                locations[d['name']].append({
                    'box': d['box'],
                    'confidence': d['confidence']
                })
    
    for object_type, boxes in locations.items():
        if len(boxes) <= 100:
            continue
        locations[object_type] = random.sample(locations[object_type], 100)
    return locations

def generate_active_hour_analytics(logs):
    active_hours_data = {o: [0]*24 for o in OBJECT_TYPES}
    for video_log in logs.values():
        for timestamp, detection in video_log:
            hour = datetime.strptime(timestamp, utils.DATETIME_FORMAT_READABLE_SECOND).hour
            objects = set([d['name'] for d in detection])
            for o in objects:
                active_hours_data[o][hour] += 1
    return active_hours_data
        


if __name__ == "__main__":
    logs = utils.get_video_logs()

    location_data = generate_location_analytics(logs)
    utils.write_location_analytics(location_data)

    active_hours_data = generate_active_hour_analytics(logs)
    utils.write_active_hour_analytics(active_hours_data)