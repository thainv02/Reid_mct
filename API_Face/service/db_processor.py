import os
import json
import time
from datetime import datetime, timezone, timedelta
from bson import ObjectId
from face_utils.save_log import Logger_Days


# Tạo logger riêng cho monitor directory
path_logs = "./logs/log_mongodb/"
if not os.path.exists(path_logs):
    os.makedirs(path_logs)
file_name = path_logs + "/log_mongo"
log_obj = Logger_Days(file_name)


# Hàm ghi log vào MongoDB
def log_to_mongo(best_match_name: str, current_datetime: datetime, collection, name_camera):
    global log_obj  # Sử dụng biến toàn cục
    try:
        new_log = {
            "WriteDate": datetime.now(timezone(timedelta(hours=7))),
            "SN": str(name_camera),
            "Pin": best_match_name,
            "AttTime": current_datetime
        }
        collection.insert_one(new_log)
        print(f"Đã thêm mới ID: {best_match_name} vào lúc {current_datetime}")
        log_obj.info(f"Đã thêm {best_match_name} vào MongoDB vào lúc {current_datetime}")

    except Exception as e:
        print(f"Lỗi khi ghi vào MongoDB: {e}")
        log_obj.error(f"Lỗi khi lưu MongoDB vào file JSON: {e}")

# Hàm lưu thông tin vào file JSON
def backup_to_json(new_log, path_logs):
    global log_obj  # Sử dụng biến toàn cục
    try:
        current_time = datetime.now()
        month_year = current_time.strftime("%m%Y")
        backup_filename = path_logs + f"/backup_db_{month_year}.json"
        
        if os.path.exists(backup_filename):
            with open(backup_filename, "r+", encoding="utf-8") as backup_file:
                try:
                    data = json.load(backup_file)
                    if not isinstance(data, list):
                        data = []
                except json.JSONDecodeError:
                    data = []
                data.append(new_log)
                backup_file.seek(0)
                json.dump(data, backup_file, ensure_ascii=False, indent=4, default=json_serializable)
                backup_file.truncate()
        else:
            with open(backup_filename, "w", encoding="utf-8") as backup_file:
                json.dump([new_log], backup_file, ensure_ascii=False, indent=4, default=json_serializable)
        
        print(f"Đã lưu backup vào {backup_filename}")
        log_obj.info(f"Đã thêm {new_log['Pin']} vào json backup vào lúc {new_log['AttTime']}")
    except Exception as e:
        print(f"Lỗi khi lưu backup vào file JSON: {e}")
        log_obj.error(f"Lỗi khi lưu backup vào file JSON: {e}")



def log_to_mongo_if_not_in_json(best_match_name: str, current_datetime: datetime, collection, name_camera, path_logs):
    global log_obj  # Sử dụng biến toàn cục
    try:
        json_file = path_logs + "/face_logs.json"
        
        # Tạo file JSON nếu chưa tồn tại
        if not os.path.exists(json_file):
            with open(json_file, "w") as f:
                json.dump({}, f)

        # Đọc dữ liệu từ file JSON
        with open(json_file, "r") as f:
            try:
                data = json.load(f)
                if not isinstance(data, dict):
                    data = {}
            except json.JSONDecodeError:
                data = {}

        # Kiểm tra nếu `best_match_name` chưa có trong JSON
        if best_match_name not in data:
            # Thêm thông tin vào MongoDB
            log_to_mongo(best_match_name, current_datetime, collection, name_camera)
            
            # Tạo log giống như khi lưu vào MongoDB
            new_log = {
                "WriteDate": datetime.now(timezone(timedelta(hours=7))),
                "SN": str(name_camera),
                "Pin": best_match_name,
                "AttTime": current_datetime
            }
            # Gọi hàm lưu vào file JSON (backup)
            backup_to_json(new_log, path_logs)

            # Thêm thông tin vào JSON
            data[best_match_name] = {
                "added_time": time.time(),
                "thoigian": current_datetime.isoformat()
            }
            
            # Gửi thông báo
            timenow = datetime.now()
            formatted_time = timenow.strftime("%H:%M:%S %d/%m/%Y")


            # Ghi lại JSON sau khi cập nhật
            with open(json_file, "w") as f:
                json.dump(data, f, default=json_serializable)

    except Exception as e:
        log_obj.error(f"Lỗi khi xử lý log_to_mongo_if_not_in_json: {e}")
        print(f"Lỗi khi xử lý log_to_mongo_if_not_in_json: {e}")

# Hàm chuyển đổi ObjectId và datetime
def json_serializable(obj):
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} is not serializable")