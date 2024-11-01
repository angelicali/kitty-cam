import json
from pathlib import Path
from datetime import datetime
import logging

DATETIME_FORMAT = '%Y%m%d%H%M%S'
DATETIME_FORMAT_READABLE = '%Y/%m/%d %H:%M'
DATETIME_FORMAT_READABLE_SECOND = '%Y/%m/%d %H:%M:%S'

VIDEO_DIR = Path('static')
VIDEO_LOG_DIR = Path('logs/byvideo/')
ANALYTICS_DIR = Path('analytics/')
ANALYTICS_LOCATION_DIR = ANALYTICS_DIR / 'location'
ANALYTICS_ACTIVE_HOUR_DIR = ANALYTICS_DIR / 'active_hour'
TRASH_DIR = Path('trash-bin')

logger = logging.getLogger(__name__)

def get_video_list(video_dir, skip_latest=False, max_videos=200):
    video_files = list(video_dir.iterdir())
    video_files.sort(reverse=True)
    if skip_latest:
        video_files = video_files[1:]
    if max_videos is not None:
        video_files = video_files[:max_videos]
    return video_files

def get_video_logs():
    logs = {}
    videos = get_video_list(VIDEO_DIR)
    for video_filename in videos:
        video_id = video_filename.split('.')[0]
        if video_id == "20241026213043":    
            break    
        video_log = get_video_log(video_id)
        logs[video_id] = video_log
    return logs

def get_video_log(video_id):
    json_logfile = VIDEO_LOG_DIR / (video_id + '.json')
    if json_logfile.exists():
        with json_logfile.open('r') as f:
            return json.loads(f.read())
    
    jsonl_logfile = VIDEO_LOG_DIR / (video_id + '.jsonl')
    if jsonl_logfile.exists():
        with jsonl_logfile.open('r') as f:
            detections = [json.loads(line) for line in f]
            return detections

def get_latest(dir):
    files = list(dir.iterdir())
    files.sort(reverse=True)
    return files[0]

def write_analytics(data, dir):
    today = str(datetime.now().date())
    output_path = dir / (today + '.json')
    with output_path.open('w') as f:
        json.dumps(data, f)

def get_analytics(dir, return_json):
    file = get_latest(dir)
    with file.open('r') as f:
        return json.loads(f.read()) if return_json else f.read()

def write_location_analytics(data):
    write_analytics(data, ANALYTICS_LOCATION_DIR)

def get_location_analytics(return_json=True):
    get_analytics(ANALYTICS_LOCATION_DIR, return_json)

def write_active_hour_analytics(data):
    write_analytics(data, ANALYTICS_ACTIVE_HOUR_DIR)

def get_active_hour_analytics(return_json=True):
    get_analytics(ANALYTICS_ACTIVE_HOUR_DIR, return_json)

def delete_video(filename):
    filename = Path(filename)
    video_id = filename.stem
    video_path = VIDEO_DIR / filename
    video_log_path = VIDEO_LOG_DIR / f"{video_id}.json"
    if video_path.exists():
        video_path.rename(TRASH_DIR / video_path.name)
        logger.info(f"File moved to trash-bin: {video_path}")
    else:
        logger.error(f"File for deletion can't be found: {video_path}")
    
    if video_log_path.eixsts():
        video_log_path.rename(TRASH_DIR / video_log_path.name)
        logger.info(f"File moved to trash-bin: {video_log_path}")
    else:
        logger.error(f"File for deletion can't be found: {video_log_path}")
