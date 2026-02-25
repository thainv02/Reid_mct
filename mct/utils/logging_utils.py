"""
Logging utilities for MCT.
"""
import os
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

from .time_utils import get_vn_time
from .file_utils import load_json_file, save_json_file


# Global loggers cache
_loggers = {}


def get_floor_logger(floor_id):
    """
    Get or create a logger for a specific floor.
    
    Args:
        floor_id: Floor identifier (e.g., 1, 3)
    
    Returns:
        logging.Logger: Configured logger for the floor
    """
    if floor_id not in _loggers:
        logger = logging.getLogger(f"Floor_{floor_id}")
        logger.setLevel(logging.INFO)
        
        # Check if handler exists to avoid duplicates
        if not logger.handlers:
            # Create handler
            log_file = f"./logs/floor_{floor_id}.log"
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            handler = RotatingFileHandler(
                log_file, 
                maxBytes=5*1024*1024,  # 5MB
                backupCount=2, 
                encoding='utf-8'
            )
            
            # Format: Timestamp - Message
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            handler.setFormatter(formatter)
            
            logger.addHandler(handler)
            logger.propagate = False  # Do not print to console
            
        _loggers[floor_id] = logger
    return _loggers[floor_id]


def remove_expired_names(log_dir, expire_seconds=60):
    """
    Remove entries older than expire_seconds from face_logs.json
    
    Args:
        log_dir: Directory containing face_logs.json
        expire_seconds: Expiration time in seconds
    
    Returns:
        dict: Updated face logs
    """
    face_logs_path = os.path.join(log_dir, "face_logs.json")
    face_logs = load_json_file(face_logs_path)
    
    current_time = get_vn_time()
    updated_logs = {}
    
    for name, data in face_logs.items():
        try:
            # Handle both old format (string) and new format (dict)
            if isinstance(data, str):
                timestamp_str = data
            elif isinstance(data, dict):
                timestamp_str = data.get('timestamp', data.get('time', ''))
            else:
                continue
                
            log_time = datetime.fromisoformat(timestamp_str)
            if (current_time - log_time).total_seconds() < expire_seconds:
                # Keep in new format
                if isinstance(data, dict):
                    updated_logs[name] = data
                else:
                    updated_logs[name] = {"timestamp": timestamp_str, "reid_id": None}
        except:
            pass
    
    save_json_file(face_logs_path, updated_logs)
    return updated_logs


def log_face_detection(name, cam_name, log_dir, person_id=None, mct_tracker=None, floor="Unknown"):
    """
    Log face detection to face_logs.json and backup_db_*.json
    
    Args:
        name: Person name (e.g., 'nv001')
        cam_name: Camera name (e.g., 'cam01')
        log_dir: Directory to store logs (e.g., './logs/cam01')
        person_id: ReID person ID (e.g., 5)
        mct_tracker: Optional MCT tracker instance for database logging
        floor: Floor name (e.g., '3F')
    
    Returns:
        bool: True if logged (new detection), False if already logged recently
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Remove expired entries
    face_logs = remove_expired_names(log_dir, expire_seconds=60)
    
    # Check if already logged in last 60 seconds
    if name in face_logs:
        return False
    
    # Get current time
    current_time = get_vn_time()
    time_str = current_time.isoformat()
    
    # Update face_logs.json (temporary log for 1 minute)
    face_logs[name] = {
        "timestamp": time_str,
        "reid_id": person_id
    }
    face_logs_path = os.path.join(log_dir, "face_logs.json")
    save_json_file(face_logs_path, face_logs)
    
    # Add to backup_db_*.json (permanent log)
    backup_filename = f"backup_db_{current_time.strftime('%m%Y')}.json"
    backup_path = os.path.join(log_dir, backup_filename)
    
    backup_data = load_json_file(backup_path)
    if not isinstance(backup_data, list):
        backup_data = []
    
    entry = {
        "WriteDate": time_str,
        "SN": cam_name,
        "Pin": name,
        "AttTime": time_str
    }
    
    # Add ReID ID if available
    if person_id is not None:
        entry["ReID_ID"] = person_id
    
    backup_data.append(entry)
    save_json_file(backup_path, backup_data)
    
    if person_id is not None:
        print(f"📝 Logged: {name} (ID:{person_id}) at {cam_name} - {current_time.strftime('%H:%M:%S')}")
        
        # MCT DB LOGGING: Face Recognition
        if mct_tracker is not None:
            try:
                mct_tracker.save_face_recognition(
                    local_track_id=person_id,
                    usr_id=name,
                    floor=floor,
                    camera_id=cam_name,
                    confidence=1.0
                )
            except Exception as e:
                print(f"Warning: Failed to log face to DB: {e}")
    else:
        print(f"📝 Logged: {name} at {cam_name} - {current_time.strftime('%H:%M:%S')}")
    
    return True
