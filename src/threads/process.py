import cv2
import time
import queue
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
from src.utils.logger import logger

class BatchProcessThread(QThread):
    fps_signal = pyqtSignal(float)  # Phát FPS của luồng Batch AI ra UI

    def __init__(self, frame_queues, tracking_queues, model_path="yolov8n.pt",
                 conf_threshold=0.25, iou_threshold=0.45, imgsz=640, device="auto"):
        super().__init__()
        self.frame_queues = frame_queues
        self.tracking_queues = tracking_queues
        
        # Tự động phát hiện GPU/CUDA hoặc sử dụng cấu hình tùy chọn
        import torch
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Đang tải mô hình YOLO duy nhất từ: {model_path} trên thiết bị: {self.device}")
        self.model = YOLO(model_path)
        try:
            self.model.to(self.device)
        except Exception as e:
            logger.warning(f"Không thể chuyển mô hình sang thiết bị {self.device}: {e}")
            
        self.running = True
        
        # Lưu trữ độc lập trạng thái hoạt động ROI của từng camera
        self.active_rois = [False] * len(frame_queues)
        self.rel_roi_points_list = [[] for _ in range(len(frame_queues))]
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self._proc_fps = 0.0
        self._last_time = time.time()
        self._last_fps_emit = 0.0

    def set_roi_active(self, cam_idx, is_active):
        if 0 <= cam_idx < len(self.active_rois):
            self.active_rois[cam_idx] = is_active

    def set_roi(self, cam_idx, rel_points):
        if 0 <= cam_idx < len(self.rel_roi_points_list):
            self.rel_roi_points_list[cam_idx] = rel_points

    def run(self):
        logger.info("BatchProcessThread bắt đầu chạy Detection gộp.")
        while self.running:
            batch_frames = []
            batch_inputs = []
            batch_cap_times = []
            active_indices = []

            # 1. Thu thập frame từ tất cả camera (phi block)
            for idx, q in enumerate(self.frame_queues):
                try:
                    # Lấy frame phi block
                    frame, cap_time = q.get_nowait()
                    
                    annotated_frame = frame.copy()
                    
                    # Nếu ROI của camera này không active, ta KHÔNG đưa vào bộ gom batch chạy YOLO
                    # mà đẩy thẳng với raw_dets rỗng xuống tracking queue để duy trì video hiển thị mượt mà
                    if idx >= len(self.active_rois) or not self.active_rois[idx]:
                        raw_dets = np.empty((0, 6), dtype=np.float32)
                        t_q = self.tracking_queues[idx]
                        if t_q.empty():
                            t_q.put((annotated_frame, raw_dets, cap_time))
                        continue
                    
                    # Nếu ROI active, chuẩn bị yolo_input
                    yolo_input = annotated_frame.copy()
                    if idx < len(self.rel_roi_points_list) and self.rel_roi_points_list[idx]:
                        h, w = yolo_input.shape[:2]
                        roi_pts = np.array(
                            [(int(p[0]*w), int(p[1]*h)) for p in self.rel_roi_points_list[idx]],
                            dtype=np.int32
                        )
                        mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.fillPoly(mask, [roi_pts], 255)
                        yolo_input = cv2.bitwise_and(yolo_input, yolo_input, mask=mask)

                    batch_frames.append(annotated_frame)
                    batch_inputs.append(yolo_input)
                    batch_cap_times.append(cap_time)
                    active_indices.append(idx)
                except queue.Empty:
                    # Camera này chưa có frame mới trong chu kỳ này
                    continue

            # Nếu không có camera nào active cần chạy YOLO, tạm dừng cực ngắn rồi lặp lại
            if not batch_inputs:
                time.sleep(0.001)
                continue

            try:
                # 2. Chạy Batch Inference bằng YOLO cho các camera đang active ROI
                results = self.model(
                    batch_inputs,
                    classes=[0],
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False
                )

                # 3. Phân phối kết quả về đúng camera tương ứng
                for i, cam_idx in enumerate(active_indices):
                    res = results[i]
                    boxes = res.boxes
                    annotated_frame = batch_frames[i]
                    
                    if boxes is not None and len(boxes) > 0:
                        raw_dets = boxes.data.cpu().numpy()  # shape (N, 6)
                        # Vẽ bbox mảnh lên frame gốc
                        for det in raw_dets:
                            x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
                    else:
                        raw_dets = np.empty((0, 6), dtype=np.float32)

                    # Đẩy sang tracking queue tương ứng của camera đó
                    t_q = self.tracking_queues[cam_idx]
                    if t_q.empty():
                        t_q.put((annotated_frame, raw_dets, batch_cap_times[i]))

            except Exception as e:
                logger.error(f"Lỗi trong Batch Inference: {e}")

            # 4. Tính và emit FPS của BatchProcessThread (mỗi 1 giây)
            now = time.time()
            diff = now - self._last_time
            if diff > 0:
                self._proc_fps = self._proc_fps * 0.9 + (1.0 / diff) * 0.1
            self._last_time = now
            if now - self._last_fps_emit >= 1.0:
                self.fps_signal.emit(self._proc_fps)
                self._last_fps_emit = now

    def stop(self):
        self.running = False
        self.wait()