import cv2
import time
import queue
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
from src.utils.logger import logger

class BatchProcessThread(QThread):
    fps_signal = pyqtSignal(list)  # Phát danh sách FPS của luồng Batch AI cho từng camera ra UI

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
        self.active_cameras = [True] * len(frame_queues)  # True = camera đang chạy, False = camera đang dừng
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        
        self.camera_frame_counts = [0] * len(frame_queues)
        self.camera_proc_fps = [0.0] * len(frame_queues)
        self._last_fps_time = time.time()

    def set_camera_active(self, cam_idx, is_active):
        if 0 <= cam_idx < len(self.active_cameras):
            self.active_cameras[cam_idx] = is_active

    def set_roi_active(self, cam_idx, is_active):
        if 0 <= cam_idx < len(self.active_rois):
            self.active_rois[cam_idx] = is_active

    def set_roi(self, cam_idx, rel_points):
        if 0 <= cam_idx < len(self.rel_roi_points_list):
            self.rel_roi_points_list[cam_idx] = rel_points

    def run(self):
        logger.info("BatchProcessThread bắt đầu chạy Detection gộp (Chủ động gom Batch + Timeout).")
        batch_timeout = 0.010  # 10ms timeout để gom các camera khác
        
        while self.running:
            # 1. Chờ đợi chủ động frame đầu tiên từ bất kỳ camera nào đang hoạt động để tránh CPU spin
            first_frame_idx = -1
            first_frame_data = None
            
            for idx, q in enumerate(self.frame_queues):
                # Bỏ qua hoàn toàn các camera đang bị tạm dừng (stop)
                if not self.active_cameras[idx]:
                    continue
                try:
                    # Chờ tối đa 5ms trên mỗi queue để kiểm tra tín hiệu
                    first_frame_data = q.get(timeout=0.005)
                    first_frame_idx = idx
                    break
                except queue.Empty:
                    continue
            
            # Nếu hết vòng lặp mà vẫn không có frame nào từ các camera active, quay lại đầu loop
            if first_frame_data is None:
                continue
                
            # 2. Đã có frame đầu tiên! Bắt đầu kích hoạt cửa sổ gom Batch (Timeout 10ms)
            batch_frames = [None] * len(self.frame_queues)
            batch_cap_times = [None] * len(self.frame_queues)
            
            # Lưu thông tin frame đầu tiên
            batch_frames[first_frame_idx], batch_cap_times[first_frame_idx] = first_frame_data
            self.camera_frame_counts[first_frame_idx] += 1
            
            start_gather = time.time()
            
            # Gom từ các camera active còn lại trong khoảng thời gian timeout
            while self.running:
                elapsed = time.time() - start_gather
                if elapsed >= batch_timeout:
                    break
                
                all_filled = True
                for idx, q in enumerate(self.frame_queues):
                    # Bỏ qua camera bị dừng
                    if not self.active_cameras[idx]:
                        continue
                    if batch_frames[idx] is None:
                        try:
                            frame, cap_time = q.get_nowait()
                            batch_frames[idx] = frame
                            batch_cap_times[idx] = cap_time
                            self.camera_frame_counts[idx] += 1
                        except queue.Empty:
                            all_filled = False
                
                if all_filled:
                    break
                
                # Nghỉ cực ngắn để nhường CPU trong lúc gom
                time.sleep(0.001)
                
            # 3. Phân loại và chuẩn bị dữ liệu cho Batch Inference
            inference_batch_frames = []
            inference_batch_inputs = []
            inference_batch_cap_times = []
            inference_active_indices = []
            
            for idx in range(len(self.frame_queues)):
                frame = batch_frames[idx]
                if frame is None:
                    continue
                    
                cap_time = batch_cap_times[idx]
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
                
                inference_batch_frames.append(annotated_frame)
                inference_batch_inputs.append(yolo_input)
                inference_batch_cap_times.append(cap_time)
                inference_active_indices.append(idx)
                
            # 4. Thực hiện Batch Inference nếu có ít nhất 1 camera active ROI
            if inference_batch_inputs:
                try:
                    results = self.model(
                        inference_batch_inputs,
                        classes=[0],
                        conf=self.conf_threshold,
                        iou=self.iou_threshold,
                        imgsz=self.imgsz,
                        device=self.device,
                        verbose=False
                    )
                    
                    # Phân phối kết quả về đúng camera tương ứng
                    for i, cam_idx in enumerate(inference_active_indices):
                        res = results[i]
                        boxes = res.boxes
                        annotated_frame = inference_batch_frames[i]
                        
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
                            t_q.put((annotated_frame, raw_dets, inference_batch_cap_times[i]))
                            
                except Exception as e:
                    logger.error(f"Lỗi trong Batch Inference: {e}")

            # 4. Tính và emit FPS của BatchProcessThread dựa trên số frame thực tế (mỗi 1 giây)
            now = time.time()
            elapsed = now - self._last_fps_time
            if elapsed >= 1.0:
                for i in range(len(self.camera_frame_counts)):
                    fps = self.camera_frame_counts[i] / elapsed
                    self.camera_proc_fps[i] = self.camera_proc_fps[i] * 0.9 + fps * 0.1
                    self.camera_frame_counts[i] = 0
                self.fps_signal.emit(self.camera_proc_fps)
                self._last_fps_time = now

    def stop(self):
        self.running = False
        self.wait()