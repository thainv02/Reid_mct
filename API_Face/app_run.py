import multiprocessing
import time
import os
import cv2
import random
import torch
import onnxruntime as ort
from pymongo import MongoClient
from load_model import load_model
from c.cConst import Const
from service.processing import build_targets
from service.frame_processor import frame_processor

from face_utils.save_log import Logger_Days, Logger_maxBytes
from face_utils.utils import count_directories_and_files
from face_utils.process_db import connect_to_mongo
from face_utils.process_camera import check_camera_connection, connect_camera


# Initialize constants and logging
var = Const()

print(f"DEBUG: Connection String: '{var.connection_string}'")
print(f"DEBUG: Torch CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"DEBUG: CUDA Device: {torch.cuda.get_device_name(0)}")
print(f"DEBUG: ONNX Runtime Providers: {ort.get_available_providers()}")

def monitor_directory(directory, queue):
    """
    Monitors directory for changes in number of files/folders.
    When changes detected, reloads model and updates targets.
    """
    # Tạo logger riêng cho monitor directory
    path_logs = "./logs/directory_monitor"
    if not os.path.exists(path_logs):
        os.makedirs(path_logs)
    file_name = path_logs + "/logs_directory_changes"
    log_obj = Logger_Days(file_name)

    initial_folders, initial_files = count_directories_and_files(directory)
    log_obj.info(f"Initial state - Folders: {initial_folders}, Files: {initial_files}")

    while True:
        current_folders, current_files = count_directories_and_files(directory)
        
        if initial_folders != current_folders or initial_files != current_files:
            log_obj.info(f"Directory change detected!")
            log_obj.info(f"Previous state - Folders: {initial_folders}, Files: {initial_files}")
            log_obj.info(f"Current state - Folders: {current_folders}, Files: {current_files}")

            try:
                # Bắt đầu quá trình cập nhật embedding
                log_obj.info("Starting model reload and target building process...")
                
                detector, recognizer = load_model()
                log_obj.info("Model loaded successfully")
                
                targets = build_targets(detector, recognizer, directory)
                log_obj.info(f"Built targets successfully. Total targets: {len(targets)}")
                
                # Đưa targets mới vào queue
                queue.put(targets)
                log_obj.info("New targets added to queue")

                # Giải phóng bộ nhớ
                del detector, recognizer
                torch.cuda.empty_cache()
                log_obj.info("GPU memory cleared")

                # notify_error("Đã cập nhật khuôn mặt mới!")

            except Exception as e:
                log_obj.info(f"Error in embedding update process: {str(e)}")
                # notify_error(f"Lỗi cập nhật nhân viên mới !: {str(e)}")
            
            # Cập nhật trạng thái ban đầu
            initial_folders, initial_files = current_folders, current_files


def process_changes(queue_embeddings, link_camera, name_camera):
    """
    Main processing function that:
    1. Connects to MongoDB
    2. Loads face detection/recognition models
    3. Processes video frames
    4. Updates targets when directory changes
    5. Monitors camera connection and attempts reconnection
    """
    path_logs = "./logs/" + str(name_camera)
    if not os.path.exists(path_logs):
        os.makedirs(path_logs)
    file_name = path_logs + "/logs_original"
    log_obj = Logger_Days(file_name)

    collection = connect_to_mongo(log_obj)
    
    detector, recognizer = load_model()
    targets = build_targets(detector, recognizer, var.faces_dir)
    colors = {name: (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)) 
             for _, name in targets}

    cap = None
    last_connection_check = time.time()
    connection_check_interval = 60  # Kiểm tra kết nối mỗi 60 giây
    count_time = 0

    while True:
        # Kiểm tra và thiết lập kết nối camera
        current_time = time.time()
        if cap is None or (current_time - last_connection_check >= connection_check_interval):
            if cap is not None:
                if not check_camera_connection(cap):
                    log_obj.info("Mất kết nối camera - Đang thử kết nối lại...")
                    cap.release()
                    cap = None
                
            if cap is None:
                cap = connect_camera(link_camera, log_obj)
                if cap is None:
                    log_obj.info("Không thể kết nối camera - Sẽ thử lại sau 5 giây:")
                    time.sleep(5)
                    count_time = count_time + 1
                    if count_time >= 10:
                        # notify_error("Không thể kết nối camera!")
                        count_time = 0
                    continue
            last_connection_check = current_time

        # Đọc và xử lý frame
        ret, frame = cap.read()
        if not ret:
            log_obj.info("Không thể đọc frame từ camera:")
            cap.release()
            cap = None
            continue

        # Xử lý cập nhật targets từ queue
        if not queue_embeddings.empty():
            print(f"Số lượng file trong thư mục đã thay đổi.")
            log_obj.info("---Số lượng file trong thư mục đã thay đổi.---")
            log_obj.info("---UPDATE TARGETS EMBEDDINGS---")
            targets = queue_embeddings.get()
            

        # Xử lý frame
        frame_count = getattr(process_changes, 'frame_count', 0)

        count_error_frame = 0

        if frame_count % var.max_frame == 0:
            try:
                frame = frame_processor(frame, detector, recognizer, targets, 
                                     colors, collection, var, name_camera, log_obj, path_logs)
                
                # Hien thi frame ket qua
                cv2.namedWindow(name_camera, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(name_camera, 640, 480) 
                cv2.imshow(name_camera, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                error_msg = f"Lỗi khi xử lý frame hình ảnh: {str(e)}"
                print(error_msg)
                log_obj.info(error_msg)
                count_error_frame = count_error_frame + 1
                if count_error_frame > 15:
                    count_error_frame = 0
                    # notify_error(f"Lỗi khi xử lý frame hình ảnh: {str(e)}")


        frame_count = (frame_count + 1) % 1000
        process_changes.frame_count = frame_count

if __name__ == "__main__":
    link_camera_01, link_camera_02, path_to_face = var.source, var.source1, var.faces_dir
    # Create queue for inter-process communication
    queue_embeddings = multiprocessing.Queue()

    # Create and start monitoring and processing processes
    monitor_process = multiprocessing.Process(target=monitor_directory, 
                                            args=(path_to_face, queue_embeddings))
    process_camera_01 = multiprocessing.Process(target=process_changes, 
                                            args=(queue_embeddings, link_camera_01, "cam01"))
    process_camera_02 = multiprocessing.Process(target=process_changes, 
                                            args=(queue_embeddings, link_camera_02, "cam02"))
    
    monitor_process.start()
    process_camera_01.start()
    process_camera_02.start()

    monitor_process.join()
    process_camera_01.join()
    process_camera_02.join()