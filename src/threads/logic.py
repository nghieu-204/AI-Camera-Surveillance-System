import cv2
import queue
import numpy as np
import time
from PyQt5.QtCore import QThread, pyqtSignal
from src.utils.logger import logger

class LogicThread(QThread):
    # Phát tín hiệu về UI: frame_đã_vẽ, số_người, có_cảnh_báo_không, fps, danh_sách_cảnh_báo, latency
    result_signal = pyqtSignal(object, int, bool, float, list, float)

    def __init__(self, logic_queue, target_ui_fps=15):
        super().__init__()
        self.logic_queue = logic_queue
        self.running = True
        self.rel_roi_points = []
        self.roi_points = []
        self.entry_times = {}
        self.dwell_threshold = 3.0
        self.crowd_threshold = 5
        
        self.last_time = time.time()
        self.fps = 0.0
        self._ui_interval = 1.0 / target_ui_fps  # Khoảng cách tối thiểu giữa 2 lần emit
        self._last_emit = 0.0
        self.paused = False

    def toggle_pause(self):
        self.paused = not self.paused
        return self.paused

    def set_roi(self, rel_points):
        self.rel_roi_points = rel_points

    def run(self):
        logger.info("LogicThread bắt đầu chạy (ROI & Warning Logic).")
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue
            try:
                frame, objects, cap_time = self.logic_queue.get(timeout=1)
                warning_triggered = False
                
                current_time = time.time()
                time_diff = current_time - self.last_time
                if time_diff > 0:
                    self.fps = (self.fps * 0.9) + ((1.0 / time_diff) * 0.1)
                self.last_time = current_time

                # --- Throttle: kiểm tra có nên emit lần này không ---
                should_emit = (current_time - self._last_emit) >= self._ui_interval

                # Nếu không có ROI và chưa đến giờ emit, bỏ qua toàn bộ xử lý nặng
                has_roi = len(self.rel_roi_points) > 0
                if not has_roi and not should_emit:
                    continue
                
                # Tạo bản sao của frame để vẽ
                annotated_frame = frame.copy()
                h, w = annotated_frame.shape[:2]
                warning_triggered = False

                # Tính toán lại toạ độ ROI
                if len(self.rel_roi_points) > 0:
                    self.roi_points = [(int(pt[0] * w), int(pt[1] * h)) for pt in self.rel_roi_points]
                else:
                    self.roi_points = []

                # Vẽ ROI lên ảnh nếu có
                pts = None
                if len(self.roi_points) >= 3:
                    pts = np.array(self.roi_points, np.int32)
                    pts_reshape = pts.reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [pts_reshape], isClosed=True, color=(255, 0, 0), thickness=2)
                # Lọc các object nằm trong ROI
                filtered_objects = {}
                for obj_id, (centroid, box) in objects.items():
                    startX, startY, endX, endY = box
                    bottom_center = ((startX + endX) // 2, endY)
                    
                    in_roi = False
                    if pts is not None:
                        if cv2.pointPolygonTest(pts, bottom_center, False) >= 0:
                            in_roi = True
                    else:
                        # Nếu không vẽ ROI, có thể coi là false hoặc true tùy logic.
                        # Ở đây tạm bỏ qua không đếm nếu không có ROI.
                        pass
                        
                    if in_roi:
                        filtered_objects[obj_id] = (centroid, box)
                
                # Cập nhật entry_times dựa trên các object trong ROI
                current_ids = list(filtered_objects.keys())
                for obj_id in current_ids:
                    if obj_id not in self.entry_times:
                        self.entry_times[obj_id] = current_time
                
                # Dọn dẹp các ID đã biến mất khỏi entry_times
                ids_to_remove = [obj_id for obj_id in self.entry_times if obj_id not in current_ids]
                for obj_id in ids_to_remove:
                    del self.entry_times[obj_id]
                
                person_count = len(filtered_objects)
                
                # Logic Cảnh báo CROWD
                if person_count >= self.crowd_threshold:
                    warning_triggered = True
                    cv2.putText(annotated_frame, "CROWD WARNING!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                # Vẽ và kiểm tra cảnh báo DWELL TIME cho các object trong ROI
                for obj_id, (centroid, box) in filtered_objects.items():
                    startX, startY, endX, endY = box
                    
                    dwell_time = current_time - self.entry_times.get(obj_id, current_time)
                    is_dwelling = dwell_time > self.dwell_threshold
                    
                    if is_dwelling:
                        warning_triggered = True
                        color = (0, 0, 255) # Đỏ nếu đứng lâu
                        text = f"ID: {obj_id} (DWELL {int(dwell_time)}s)"
                    else:
                        color = (0, 255, 0) # Xanh lá bình thường
                        text = f"ID: {obj_id}"
                        
                    # Vẽ Bounding Box và ID
                    cv2.rectangle(annotated_frame, (startX, startY), (endX, endY), color, 2)
                    cv2.putText(annotated_frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.circle(annotated_frame, centroid, 4, color, -1)

                # Thu thập danh sách event cảnh báo để gửi về UI panel
                alert_events = []
                if person_count >= self.crowd_threshold:
                    alert_events.append(f"🚨 [CAM {getattr(self, 'camera_id', '?')}] ĐÁM ĐÔNG: {person_count} người trong ROI")
                
                for obj_id, (centroid, box) in filtered_objects.items():
                    dwell_time = current_time - self.entry_times.get(obj_id, current_time)
                    if dwell_time > self.dwell_threshold:
                        alert_events.append(f"⚠️ [CAM {getattr(self, 'camera_id', '?')}] ID {obj_id} đứng lâu {int(dwell_time)}s trong ROI")

                # Gửi ảnh đã vẽ về UI (throttle theo target_ui_fps)
                if should_emit:
                    latency = current_time - cap_time
                    self._last_emit = current_time
                    self.result_signal.emit(annotated_frame, person_count, warning_triggered, self.fps, alert_events, latency)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Lỗi trong LogicThread: {e}")

    def stop(self):
        self.running = False
        self.wait()
