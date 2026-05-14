import cv2
import time
import queue
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
from src.utils.logger import logger

class ProcessThread(QThread):
    fps_signal = pyqtSignal(float)  # Phát FPS của luồng Process ra UI

    def __init__(self, frame_queue, tracking_queue, model_path="yolov8n.pt",
                 conf_threshold=0.25, iou_threshold=0.45, imgsz=640):
        super().__init__()
        self.frame_queue = frame_queue
        self.tracking_queue = tracking_queue
        logger.info(f"Đang tải mô hình YOLO từ: {model_path}")
        self.model = YOLO(model_path)
        self.running = True
        self.active_roi = False
        self.rel_roi_points = []
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self._proc_fps = 0.0
        self._last_time = time.time()
        self._last_fps_emit = 0.0

    def set_roi_active(self, is_active):
        self.active_roi = is_active

    def set_roi(self, rel_points):
        """Nhận ROI từ UI để mask frame trước khi đưa vào YOLO."""
        self.rel_roi_points = rel_points

    def run(self):
        logger.info("ProcessThread bắt đầu chạy Detection.")
        while self.running:
            try:
                # Lấy frame từ hàng đợi (không chờ đợi quá lâu)
                frame = self.frame_queue.get(timeout=1)
                
                annotated_frame = frame.copy()
                
                if self.active_roi:
                    # Tạo mask: chỉ giữ lại vùng ROI cho YOLO
                    yolo_input = annotated_frame.copy()
                    if self.rel_roi_points:
                        h, w = yolo_input.shape[:2]
                        roi_pts = np.array(
                            [(int(p[0]*w), int(p[1]*h)) for p in self.rel_roi_points],
                            dtype=np.int32
                        )
                        mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.fillPoly(mask, [roi_pts], 255)
                        yolo_input = cv2.bitwise_and(yolo_input, yolo_input, mask=mask)

                    # Chỉ detect thuần — không tracking, không ID
                    results = self.model(
                        yolo_input,
                        classes=[0],
                        conf=self.conf_threshold,
                        iou=self.iou_threshold,
                        imgsz=self.imgsz,
                        verbose=False
                    )

                    # Trích xuất raw detections dưới dạng numpy (N×6): [x1,y1,x2,y2,conf,cls]
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        raw_dets = boxes.data.cpu().numpy()  # shape (N, 6)
                        # Vẽ bbox mảnh lên frame gốc
                        for det in raw_dets:
                            x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
                    else:
                        raw_dets = np.empty((0, 6), dtype=np.float32)
                else:
                    raw_dets = np.empty((0, 6), dtype=np.float32)

                # Tính và emit FPS của ProcessThread (mỗi 1 giây)
                now = time.time()
                diff = now - self._last_time
                if diff > 0:
                    self._proc_fps = self._proc_fps * 0.9 + (1.0 / diff) * 0.1
                self._last_time = now
                if now - self._last_fps_emit >= 1.0:
                    self.fps_signal.emit(self._proc_fps)
                    self._last_fps_emit = now

                # Đẩy sang luồng Tracking
                if self.tracking_queue.empty():
                    self.tracking_queue.put((annotated_frame, raw_dets))
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Lỗi trong ProcessThread: {e}")

    def stop(self):
        self.running = False
        self.wait()