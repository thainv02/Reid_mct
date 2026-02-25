"""
Display Manager for MCT System.
Manages cv2.imshow windows for floor maps and camera feeds.
Only shows displays for floors that are enabled via the API.
"""
import cv2
import threading

# Track which floors are currently being displayed
_display_lock = threading.Lock()
_active_display_floors = set()  # {floor_num, ...}


def update_map_display(floor_num: int, map_img):
    """
    Show/update the map window for an enabled floor.
    
    Args:
        floor_num: Floor number
        map_img: Rendered map image (numpy array)
    """
    if map_img is None:
        return
    with _display_lock:
        _active_display_floors.add(floor_num)
    
    window_name = f"Map Floor {floor_num}"
    # Resize for display if too large
    h, w = map_img.shape[:2]
    max_h = 800
    if h > max_h:
        scale = max_h / h
        display_img = cv2.resize(map_img, (int(w * scale), int(h * scale)))
    else:
        display_img = map_img
    cv2.imshow(window_name, display_img)


def show_camera_frame(cam_id: str, floor_num: int, frame, cam_name: str = None):
    """
    Show a camera feed window for an enabled camera.
    
    Args:
        cam_id: Camera ID
        floor_num: Floor number the camera belongs to
        frame: Camera frame (numpy array)
        cam_name: Optional display name for the camera
    """
    if frame is None:
        return
    
    display_name = cam_name or cam_id
    window_name = f"F{floor_num} - {display_name}"
    
    # Resize for display
    h, w = frame.shape[:2]
    max_w = 640
    if w > max_w:
        scale = max_w / w
        display_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    else:
        display_frame = frame
    cv2.imshow(window_name, display_frame)


def close_floor_display(floor_num: int, camera_ids: list = None):
    """
    Close all display windows for a disabled floor.
    
    Args:
        floor_num: Floor number to close displays for
        camera_ids: List of camera IDs on this floor
    """
    with _display_lock:
        _active_display_floors.discard(floor_num)
    
    # Close map window
    try:
        cv2.destroyWindow(f"Map Floor {floor_num}")
    except Exception:
        pass
    
    # Close camera windows
    if camera_ids:
        for cam_id in camera_ids:
            try:
                # Try both possible window name formats
                cv2.destroyWindow(f"F{floor_num} - {cam_id}")
            except Exception:
                pass


def is_floor_displayed(floor_num: int) -> bool:
    """Check if a floor is currently being displayed."""
    with _display_lock:
        return floor_num in _active_display_floors


def close_all_displays():
    """Close all display windows."""
    with _display_lock:
        _active_display_floors.clear()
    cv2.destroyAllWindows()
