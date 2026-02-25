# 📊 BÁO CÁO CẤU TRÚC DATABASE
## Hệ thống Multi-Camera Tracking (MCT)

---

**Ngày báo cáo:** 06/02/2026  
**Dự án:** TransReID / MCT  
**Mục đích:** Theo dõi và phân tích di chuyển nhân viên trong tòa nhà  

---

## 1. TỔNG QUAN

Hệ thống MCT sử dụng **3 bảng chính** trong PostgreSQL để lưu trữ dữ liệu tracking:

| STT | Tên Bảng | Mục Đích | Số Bản Ghi |
|-----|----------|----------|------------|
| 1 | `mct_sessions` | Quản lý phiên làm việc | 9 |
| 2 | `mct_face_recognition` | Lưu sự kiện nhận diện khuôn mặt | 70 |
| 3 | `mct_position_tracking` | Lưu vị trí di chuyển | 11,640 |

---

## 2. CHI TIẾT CÁC BẢNG

### 2.1. Bảng `mct_sessions` - Quản Lý Phiên Làm Việc

**Mô tả:** Mỗi lần khởi chạy hệ thống tracking sẽ tạo một session mới. Bảng này quản lý thông tin các phiên làm việc.

| Tên Trường | Kiểu Dữ Liệu | Bắt Buộc | Mô Tả |
|------------|--------------|----------|-------|
| `id` | SERIAL | ✅ | Khóa chính, tự động tăng |
| `session_id` | VARCHAR(50) | ✅ | Mã UUID định danh phiên (VD: `0130fcd6`) |
| `started_at` | TIMESTAMP WITH TIME ZONE | ✅ | Thời điểm bắt đầu phiên |
| `ended_at` | TIMESTAMP WITH TIME ZONE | ❌ | Thời điểm kết thúc (NULL nếu đang chạy) |
| `status` | VARCHAR(20) | ❌ | Trạng thái: `active` / `stopped` / `crashed` |
| `total_tracks` | INTEGER | ❌ | Tổng số người đã tracking trong phiên |
| `total_identified` | INTEGER | ❌ | Số người đã được nhận diện khuôn mặt |

**Index đã tạo:**
- `idx_mct_sessions_status` - Tối ưu truy vấn theo trạng thái
- `idx_mct_sessions_started_at` - Tối ưu truy vấn theo thời gian

---

### 2.2. Bảng `mct_face_recognition` - Nhận Diện Khuôn Mặt

**Mô tả:** Lưu trữ các sự kiện nhận diện khuôn mặt thành công. Mỗi khi hệ thống xác định được danh tính một người, thông tin sẽ được ghi vào bảng này.

| Tên Trường | Kiểu Dữ Liệu | Bắt Buộc | Mô Tả |
|------------|--------------|----------|-------|
| `id` | SERIAL | ✅ | Khóa chính, tự động tăng |
| `session_id` | VARCHAR(50) | ✅ | Liên kết đến phiên làm việc |
| `local_track_id` | INTEGER | ✅ | ID tracking nội bộ trong phiên (0, 1, 2...) |
| `usr_id` | VARCHAR(50) | ✅ | **Mã nhân viên** (VD: `INF2503004`) hoặc `unknown` |
| `floor` | VARCHAR(10) | ✅ | Tầng ghi nhận (VD: `1F`, `3F`, `7F`) |
| `camera_id` | VARCHAR(50) | ❌ | Tên camera ghi nhận (VD: `3F-IN Giữa sàn`) |
| `detected_at` | TIMESTAMP WITH TIME ZONE | ✅ | Thời điểm nhận diện |
| `confidence` | DOUBLE PRECISION | ❌ | Độ tin cậy của kết quả nhận diện (0.0 - 1.0) |
| `created_at` | TIMESTAMP WITH TIME ZONE | ❌ | Thời điểm ghi vào database |

**Index đã tạo:**
- `idx_mct_face_usr_id` - Truy vấn nhanh theo mã nhân viên
- `idx_mct_face_detected_at` - Truy vấn theo thời gian
- `idx_mct_face_session` - Truy vấn theo session và track_id
- `idx_mct_face_floor` - Truy vấn theo tầng

**Thống kê hiện tại:**
- Tổng số bản ghi: **70**
- Số nhân viên đã nhận diện: **39 người**
- Số session có dữ liệu: **2**

---

### 2.3. Bảng `mct_position_tracking` - Theo Dõi Vị Trí

**Mô tả:** Lưu trữ tọa độ vị trí của người được tracking. Dữ liệu được ghi định kỳ mỗi ~5 giây cho mỗi người đang được theo dõi.

| Tên Trường | Kiểu Dữ Liệu | Bắt Buộc | Mô Tả |
|------------|--------------|----------|-------|
| `id` | SERIAL | ✅ | Khóa chính, tự động tăng |
| `session_id` | VARCHAR(50) | ✅ | Liên kết đến phiên làm việc |
| `local_track_id` | INTEGER | ✅ | ID tracking nội bộ trong phiên |
| `usr_id` | VARCHAR(50) | ✅ | **Mã nhân viên** hoặc `unknown` |
| `floor` | VARCHAR(10) | ✅ | Tầng (VD: `1F`, `3F`, `7F`) |
| `x` | DOUBLE PRECISION | ✅ | Tọa độ X trên bản đồ tầng (đơn vị: mm) |
| `y` | DOUBLE PRECISION | ✅ | Tọa độ Y trên bản đồ tầng (đơn vị: mm) |
| `camera_id` | VARCHAR(50) | ❌ | ID camera ghi nhận (VD: `cam36`, `cam39`) |
| `bbox_center_x` | INTEGER | ❌ | Tọa độ X tâm bounding box trong frame camera (pixel) |
| `bbox_center_y` | INTEGER | ❌ | Tọa độ Y tâm bounding box trong frame camera (pixel) |
| `tracked_at` | TIMESTAMP WITH TIME ZONE | ✅ | Thời điểm ghi nhận vị trí |
| `created_at` | TIMESTAMP WITH TIME ZONE | ❌ | Thời điểm ghi vào database |

**Index đã tạo:**
- `idx_mct_pos_usr_id` - Truy vấn nhanh theo mã nhân viên
- `idx_mct_pos_tracked_at` - Truy vấn theo thời gian
- `idx_mct_pos_floor` - Truy vấn theo tầng
- `idx_mct_pos_session` - Truy vấn theo session và track_id
- `idx_mct_pos_usr_date` - **Tối ưu truy vấn lịch sử di chuyển hàng ngày**

**Thống kê hiện tại:**
- Tổng số bản ghi: **11,640**
- Số người theo dõi được: **42 người**
- Số tầng có dữ liệu: **2 tầng**

---

## 3. MỐI QUAN HỆ GIỮA CÁC BẢNG

```
┌─────────────────┐
│  mct_sessions   │
│  (Phiên làm việc)│
└────────┬────────┘
         │ session_id
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌─────────────────┐    ┌─────────────────────┐
│mct_face_recognition│    │ mct_position_tracking │
│ (Nhận diện mặt) │    │   (Vị trí di chuyển)  │
└─────────────────┘    └─────────────────────┘
         │                        │
         └──────────┬─────────────┘
                    │ usr_id
                    ▼
            ┌───────────────┐
            │  Mã nhân viên  │
            │ (Khóa liên kết)│
            └───────────────┘
```

**Ghi chú:**
- `session_id`: Liên kết dữ liệu trong cùng một phiên tracking
- `usr_id`: **Khóa quan trọng** để gộp dữ liệu từ nhiều session khác nhau

---

## 4. CÁC TRƯỜNG HỢP SỬ DỤNG

### 4.1. Xem lịch sử di chuyển của một nhân viên trong ngày

```sql
SELECT floor, x, y, camera_id, tracked_at
FROM mct_position_tracking
WHERE usr_id = 'INF1901002'
  AND DATE(tracked_at) = CURRENT_DATE
ORDER BY tracked_at;
```

### 4.2. Xem tất cả nhân viên đã được nhận diện hôm nay

```sql
SELECT usr_id, floor, camera_id, detected_at, confidence
FROM mct_face_recognition
WHERE DATE(detected_at) = CURRENT_DATE
  AND usr_id != 'unknown'
ORDER BY detected_at DESC;
```

### 4.3. Thống kê số người theo tầng

```sql
SELECT floor, COUNT(DISTINCT usr_id) as so_nguoi
FROM mct_position_tracking
WHERE DATE(tracked_at) = CURRENT_DATE
GROUP BY floor;
```

---

## 5. ĐÁNH GIÁ VÀ KHUYẾN NGHỊ

### ✅ Ưu điểm:
1. **Thiết kế tối ưu** - Index được tạo đầy đủ cho các truy vấn phổ biến
2. **Dữ liệu liên kết** - Sử dụng `usr_id` để gộp data từ nhiều nguồn
3. **Timezone chuẩn** - Sử dụng múi giờ Asia/Ho_Chi_Minh

### 📋 Khuyến nghị:
1. Cần cập nhật `ended_at` khi session kết thúc
2. Xem xét thêm partition theo thời gian cho bảng `mct_position_tracking` khi dữ liệu lớn
3. Cân nhắc thêm bảng audit log để theo dõi thay đổi

---

## 6. KẾT LUẬN

Hệ thống MCT đã được thiết kế với cấu trúc database hợp lý, đáp ứng được yêu cầu:
- ✅ Theo dõi vị trí nhân viên real-time
- ✅ Nhận diện khuôn mặt và liên kết với mã nhân viên
- ✅ Truy vấn lịch sử di chuyển theo ngày
- ✅ Hỗ trợ gộp dữ liệu từ nhiều phiên làm việc

---

**Người lập báo cáo:** AI Assistant  
**Ngày:** 06/02/2026
