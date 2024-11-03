import json
from pathlib import Path
from datetime import datetime
import logging
import ffmpeg

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

def get_video_list(skip_latest=False, max_videos=200, return_id=False):
    video_files = list(VIDEO_DIR.iterdir())
    video_files.sort(reverse=True)
    if skip_latest:
        video_files = video_files[1:]
    if max_videos is not None:
        video_files = video_files[:max_videos]
    if return_id:
        return [f.stem for f in video_files]
    else:
        return video_files

def get_video_logs():
    logs = {}
    videos = get_video_list()
    for video_filename in videos:
        video_id = video_filename.stem
        if video_id == "20241026213043":    
            break    
        video_log = get_video_log(video_id)
        if video_log is None:
            continue
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
        json.dump(data, f)

def get_analytics(dir, return_json):
    file = get_latest(dir)
    with file.open('r') as f:
        return json.loads(f.read()) if return_json else f.read()

def write_location_analytics(data):
    write_analytics(data, ANALYTICS_LOCATION_DIR)

def get_location_analytics(return_json=True):
    return get_analytics(ANALYTICS_LOCATION_DIR, return_json)

def write_active_hour_analytics(data):
    write_analytics(data, ANALYTICS_ACTIVE_HOUR_DIR)

def get_active_hour_analytics(return_json=True):
    return get_analytics(ANALYTICS_ACTIVE_HOUR_DIR, return_json)

def delete_video(filename):
    p = Path(filename)
    delete_video_by_id(p.stem)

def delete_video_by_id(video_id):
    video_path = VIDEO_DIR / f"{video_id}.mp4"
    video_log_path = VIDEO_LOG_DIR / f"{video_id}.json"
    video_jsonl_log_path = VIDEO_LOG_DIR / f"{video_id}.jsonl"
    if video_path.exists():
        video_path.rename(TRASH_DIR / video_path.name)
        logger.info(f"File moved to trash-bin: {video_path}")
    else:
        logger.error(f"File for deletion can't be found: {video_path}")
    
    if video_log_path.exists():
        video_log_path.rename(TRASH_DIR / video_log_path.name)
        logger.info(f"File moved to trash-bin: {video_log_path}")
    elif video_jsonl_log_path.exists():
        video_jsonl_log_path.rename(TRASH_DIR / video_jsonl_log_path.name)
    else:
        logger.error(f"File for deletion can't be found: {video_log_path}")

def get_video_path(video_id):
    return VIDEO_DIR / (video_id + '.mp4')

def get_video_log_path(video_id, jsonl=True):
    suffix = '.jsonl' if jsonl else '.json'
    return VIDEO_LOG_DIR / (video_id + suffix)

def merge(video_ids):
    video_ids.sort()
    new_video_id = video_ids[0]
    
    # Merge videos
    filelist_name = f'merge_filelist_{new_video_id}.txt'
    new_video_filename = VIDEO_DIR / f"{new_video_id}_new.mp4"
    with open(filelist_name, 'w') as f:
        for video in video_ids:
            f.write(f"file 'static/{video}.mp4'\n")
    ffmpeg.input(filelist_name, format='concat', safe=0).output(str(new_video_filename), c='copy').run()

    # Merge video logs
    new_log_file = VIDEO_LOG_DIR / (new_video_id + '_new.jsonl')
    with new_log_file.open('w') as wf:
        for vid in video_ids:
            file = VIDEO_LOG_DIR / (vid + '.jsonl')
            with file.open('r') as rf:
                wf.write(rf.read())

    # Delete old videos and logs
    for vid in video_ids:
        delete_video_by_id(vid)
    
    # Move new video and log to proper paths
    new_video_filename.rename(get_video_path(new_video_id))
    new_log_file.rename(get_video_log_path(new_video_id))
