import cv2

def check_camera_connection(cap):
    """Kiểm tra kết nối camera."""
    if not cap.isOpened():
        return False
    ret, frame = cap.read()
    return ret and frame is not None

def connect_camera(link_camera, log_obj):
    """Thử kết nối với camera và trả về đối tượng VideoCapture."""
    cap = cv2.VideoCapture(link_camera)
    if check_camera_connection(cap):
        log_obj.info("Kết nối camera thành công")
        return cap
    else:
        log_obj.info("Không thể kết nối tới camera")
        return None