# BÁO CÁO DỰ ÁN: HỆ THỐNG AI CAMERA GIÁM SÁT THÔNG MINH

---

## 1. Thông tin chung

| Mục | Thông tin |
|---|---|
| Tên dự án | AI Camera Surveillance System |
| Ngôn ngữ lập trình | Python 3.x |
| Nền tảng chạy | Windows (CPU) |
| Mô hình AI | Ultralytics YOLOv8 Nano (`yolov8n.pt`) |
| Thuật toán Tracking | **ByteTrack** (phân loại detect cao/thấp) |
| Cấu hình | Quản lý tập trung qua `config.yaml` |
| Giao diện người dùng | PyQt5 |
| Số lượng camera | 4 camera đồng thời |

---

## 2. Giới thiệu tổng quan

Hệ thống **AI Camera Giám sát Thông minh** là một ứng dụng desktop được xây dựng bằng Python, có khả năng xử lý đồng thời **4 luồng camera** theo thời gian thực. Hệ thống sử dụng mô hình học sâu YOLOv8 để phát hiện người, kết hợp với thuật toán theo dõi đối tượng (Object Tracking) và phân tích vùng giám sát (ROI) để sinh ra các cảnh báo an ninh tự động.

Mục tiêu của hệ thống:
- **Tự động hóa giám sát an ninh** mà không cần con người liên tục quan sát màn hình.
- **Phát hiện và cảnh báo ngay lập tức** khi có đối tượng xâm nhập vùng cấm hoặc tụ tập đông người.
- **Chạy ổn định trên phần cứng CPU thông thường** mà không yêu cầu GPU chuyên dụng.

---

## 3. Kiến trúc hệ thống

### 3.1. Kiến trúc tổng thể

Hệ thống được thiết kế theo mô hình **Pipeline Đa luồng (Multi-threaded Pipeline)**. Mỗi camera được cấp phát một pipeline xử lý riêng gồm **4 luồng (threads)** hoạt động song song và độc lập với nhau.

```
┌─────────────┐    Queue 1     ┌─────────────────┐    Queue 2     ┌──────────────────┐    Queue 3     ┌──────────────────┐
│  Capture    │ ─────────────► │    Process      │ ─────────────► │    Tracking      │ ─────────────► │     Logic        │ ──► UI
│  Thread     │  (frame raw)   │   Thread (YOLO) │  (raw dets)    │  (ByteTrack)     │  (frame+objs)  │ (Alerts + UI)    │   Signal
└─────────────┘                └─────────────────┘                └──────────────────┘                └──────────────────┘
```

Với 4 camera đang chạy đồng thời:
- **Tổng số luồng xử lý:** 4 camera × 4 luồng = **16 luồng ngầm**
- **Luồng giao diện chính (Main UI Thread):** 1 luồng quản lý cửa sổ và sự kiện người dùng

### 3.2. Chi tiết các luồng xử lý

#### Luồng 1 — `CaptureThread` (Thu thập ảnh)

| Thuộc tính | Chi tiết |
|---|---|
| File | `src/threads/capture.py` |
| Nhiệm vụ | Đọc frame từ file video hoặc camera IP (RTSP) |
| Công nghệ | OpenCV `cv2.VideoCapture` |
| Tính năng đặc biệt | Cơ chế **Reconnect tự động**: nếu mất tín hiệu, thread tự động thử kết nối lại sau 2 giây |
| Cơ chế đồng bộ | Queue backpressure (`maxsize=2`): Cap FPS tự điều chỉnh bằng tốc độ xử lý của YOLO |
| Output | Frame thô đẩy vào `frame_queue` |

#### Luồng 2 — `ProcessThread` (Phát hiện AI)

| Thuộc tính | Chi tiết |
|---|---|
| File | `src/threads/process.py` |
| Nhiệm vụ | **Chỉ chạy Detection**: YOLOv8 phát hiện người thô |
| Tối ưu hiệu năng | `imgsz=640` (mặc định) hoặc `320` (tùy chỉnh trong config) |
| ROI Masking | Che đen vùng ngoài ROI giúp YOLO chỉ tập trung xử lý vùng quan trọng |
| Output | `(annotated_frame, raw_detections)` đẩy vào `tracking_queue` |

#### Luồng 3 — `TrackingThread` (Theo dõi đối tượng - ByteTrack)

| Thuộc tính | Chi tiết |
|---|---|
| File | `src/threads/tracking.py` |
| Nhiệm vụ | Nhận detection thô, chạy **ByteTrack** để gán ID |
| Công nghệ | `ultralytics.trackers.byte_tracker.BYTETracker` |
| Đặc điểm | Dùng Kalman Filter dự đoán và xử lý cả detect confidence thấp (che khuất) |
| Output | `(frame_with_trajectory, objects_dict)` đẩy vào `logic_queue` |

#### Luồng 4 — `LogicThread` (Logic nghiệp vụ & Cảnh báo)

| Thuộc tính | Chi tiết |
|---|---|
| File | `src/threads/logic.py` |
| Nhiệm vụ | Tính toán ROI, lọc đối tượng, sinh cảnh báo, vẽ kết quả và gửi về UI |
| Tối ưu UI | Throttle `result_signal` ở tối đa **15 FPS** để tránh quá tải UI thread |
| Tối ưu CPU | Nếu **không có ROI**, bỏ qua `frame.copy()` và mọi tính toán nặng, chỉ emit khi đến giờ |
| Thuật toán ROI | `cv2.pointPolygonTest` — kiểm tra xem điểm chân (bottom center) của người có nằm trong đa giác ROI không |
| Cảnh báo Dwell | Nếu một ID đứng trong ROI > **3 giây** → đổi bbox sang màu đỏ, hiện text `DWELL Xs` |
| Cảnh báo Crowd | Nếu số người trong ROI >= **5 người** → hiện `CROWD WARNING!` |
| Output | `result_signal` → Main UI Thread (frame, count, warning, fps, alert_events) |

---

## 4. Giao diện người dùng (UI)

Giao diện được xây dựng trên nền tảng **PyQt5** với bố cục 2 cột:

```
┌──────────────────────────────────┬─────────────────────┐
│         [CAM 1]   [CAM 2]        │  🔔 CẢNH BÁO        │
│         [CAM 3]   [CAM 4]        │  REALTIME           │
│                                  │  ─────────────────  │
│  Mỗi camera widget có:           │  [08:31:05]         │
│  - Video display                 │  ⚠️ CAM1 ID2 đứng   │
│  - Nút Play/Pause                │  lâu 5s trong ROI  │
│  - Nút Draw ROI                  │  ─────────────────  │
│  - Nút Clear ROI                 │  🚨 CAM3 ĐÁM ĐÔNG  │
│  - Status bar: FPS, People,      │  6 người trong ROI │
│    ROI status                    │  ─────────────────  │
│                                  │  [🗑 Xóa cảnh báo] │
└──────────────────────────────────┴─────────────────────┘
```

### Tính năng UI:
- **Draw ROI:** Người dùng kéo chuột vẽ vùng hình chữ nhật trực tiếp lên video. ROI được lưu dưới dạng tọa độ tỉ lệ (relative coordinates) để tái tính toán chính xác khi video thay đổi kích thước.
- **Clear ROI:** Xóa vùng giám sát và tắt YOLO inference cho camera đó (tiết kiệm CPU).
- **Panel cảnh báo realtime:** Danh sách sự kiện cảnh báo từ tất cả 4 camera, tô màu đỏ cho "Đám đông" và vàng cho "Đứng lâu", kèm timestamp.
- **Hệ thống Log:** Mọi sự kiện quan trọng được ghi vào file log tại `logs/system_YYYY-MM-DD.log`.

---

## 5. Công nghệ và thư viện

| Thư viện | Phiên bản | Mục đích |
|---|---|---|
| `opencv-python` | Latest | Xử lý ảnh, vẽ, đọc video |
| `PyQt5` | Latest | Giao diện desktop |
| `ultralytics` | Latest | YOLOv8 framework |
| `torch` (CPU) | 2.12.0+cpu | Backend deep learning |
| `numpy` | Latest | Tính toán mảng số |
| `lapx` | Latest | Hỗ trợ assignment algorithm |

---

## 6. Cấu trúc thư mục

```
AI_Camera_Project/
├── main.py                          # Điểm khởi chạy ứng dụng
├── config.yaml                      # TẤT CẢ CẤU HÌNH (YOLO, Camera, UI, ROI)
├── requirements.txt                 # Danh sách thư viện
├── yolov8n.pt                       # Model weights YOLOv8 Nano
├── REPORT.md                        # Báo cáo dự án
├── src/
│   ├── threads/
│   │   ├── capture.py               # Capture (Đọc video + Reconnect)
│   │   ├── process.py               # Process (Detect thô YOLO)
│   │   ├── tracking.py              # Tracking (ByteTrack + Trajectory)
│   │   └── logic.py                 # Logic (ROI + Alerts + UI Emit)
│   ├── ui/
│   │   └── main_window.py           # UI (PyQt5)
│   └── utils/
│       ├── config_loader.py         # Module đọc config.yaml
│       └── logger.py                # Logging system
├── assets/
│   ├── test.mp4                     # Video mẫu
│   └── bytetrack.yaml               # Config ngưỡng ByteTrack
└── logs/
    └── system_YYYY-MM-DD.log        # File log hệ thống
```

---

## 7. Các tối ưu kỹ thuật quan trọng

### 7.1. Queue Backpressure (Đồng bộ tốc độ tự nhiên)
Thay vì dùng timer cứng, các queue `maxsize=2` đóng vai trò điều tiết tốc độ:
- Khi YOLO xử lý chậm → queue đầy → CaptureThread tự động drop frame và đợi
- **Kết quả:** `Cap FPS ≈ Proc FPS` — hai chỉ số tự điều chỉnh bằng tốc độ thực của YOLO

### 7.2. ROI Masking trước YOLO
Thay vì chạy YOLO trên toàn frame (1920×1080) rồi lọc kết quả, hệ thống:
1. Tạo mask đen toàn màn hình
2. Tô trắng chỉ vùng ROI bằng `cv2.fillPoly`
3. Apply mask: `cv2.bitwise_and` → YOLO chỉ "nhìn thấy" phần bên trong ROI
- **Kết quả:** Giảm false positive, YOLO tập trung tính toán đúng vùng cần thiết

### 7.3. Giảm kích thước đầu vào YOLO (`imgsz=320`)
- YOLOv8 mặc định xử lý ảnh 640×640 → ~400ms/frame trên CPU
- Với `imgsz=320` → ~100ms/frame trên CPU (**nhanh hơn ~4 lần**)
- Đánh đổi nhỏ về độ chính xác với người đứng xa, nhưng chấp nhận được cho bài toán giám sát

### 7.4. UI Throttle trong LogicThread
LogicThread emit `result_signal` tối đa 15 lần/giây (thay vì mỗi frame). Khi không có ROI, toàn bộ phần xử lý nặng (`frame.copy()`, tính toán) được bỏ qua hoàn toàn.

### 7.5. Reconnect tự động
CaptureThread xử lý mất kết nối một cách graceful:
```
Mất tín hiệu → Release cap → Sleep 2s → Thử mở lại VideoCapture → Tiếp tục
```
Không cần khởi động lại ứng dụng khi camera bị ngắt tạm thời.

---

## 8. Thuật toán Tracking: ByteTrack

**Lý do nâng cấp:** Khắc phục nhược điểm của Centroid Tracker (hay bị swap ID khi người đi cắt chéo nhau hoặc bị che khuất).

**Cơ chế hoạt động:**
1. **Kalman Filter**: Dự đoán vị trí của bbox ở frame tiếp theo dựa trên vận tốc hiện tại.
2. **IoU Matching**: So sánh bbox dự đoán với bbox thực tế bằng chỉ số IoU thay vì khoảng cách tâm.
3. **Double Association**:
   - Vòng 1: Khớp các detection điểm cao (High Conf) với các track hiện có.
   - Vòng 2: Khớp các detection điểm thấp (Low Conf - thường là người bị che khuất) với các track đang bị mất.
4. **Kết quả**: ID cực kỳ ổn định, giảm thiểu tối đa hiện tượng nhảy ID (ID switching).

---

## 9. Kết luận

Dự án đã hoàn thiện được một hệ thống giám sát AI hoàn chỉnh, hoạt động ổn định trên phần cứng CPU phổ thông với đầy đủ các tính năng:

✅ Phát hiện người theo thời gian thực (YOLOv8)  
✅ Theo dõi đối tượng và vẽ quỹ đạo (Centroid Tracker)  
✅ Giám sát theo vùng ROI tùy chỉnh  
✅ Cảnh báo đám đông và đứng lâu  
✅ Xử lý đồng thời 4 camera  
✅ Panel cảnh báo realtime tích hợp trong UI  
✅ Tự động reconnect khi mất tín hiệu camera  
✅ Tối ưu hiệu năng: Queue backpressure, ROI Masking, imgsz=320  

**Hướng phát triển tiếp theo:**
- Thay Centroid Tracker bằng **SORT (Kalman Filter)** hoặc **ByteTrack** để giảm ID switch khi người che khuất nhau
- Lưu ảnh chụp màn hình khi có cảnh báo vào thư mục `snapshots/`
- Ghi sự kiện cảnh báo vào cơ sở dữ liệu SQLite để tạo báo cáo thống kê theo ngày
- Hỗ trợ camera IP thực qua giao thức RTSP
- Tối ưu thêm bằng **OpenVINO** hoặc **ONNX Runtime** để tăng tốc YOLO trên CPU Intel
