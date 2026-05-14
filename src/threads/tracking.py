import cv2
import time
import queue
import yaml
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics.trackers.byte_tracker import BYTETracker
from src.utils.logger import logger


class DetectionResult:
    """
    Wrapper hỗ trợ numpy boolean indexing để BYTETracker.update() có thể lọc
    detections theo mảng boolean: results[mask]
    Các thuộc tính cần thiết: .xyxy, .xywh, .conf, .cls
    """
    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls  = np.asarray(cls,  dtype=np.float32).reshape(-1)
        # Tính xywh (cx, cy, w, h) từ xyxy (x1, y1, x2, y2)
        cx = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2
        cy = (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2
        w  = self.xyxy[:, 2] - self.xyxy[:, 0]
        h  = self.xyxy[:, 3] - self.xyxy[:, 1]
        self.xywh = np.stack([cx, cy, w, h], axis=1)

    def __getitem__(self, idx):
        result = DetectionResult.__new__(DetectionResult)
        result.xyxy = self.xyxy[idx]
        result.conf = self.conf[idx]
        result.cls  = self.cls[idx]
        cx = (result.xyxy[:, 0] + result.xyxy[:, 2]) / 2
        cy = (result.xyxy[:, 1] + result.xyxy[:, 3]) / 2
        w  = result.xyxy[:, 2] - result.xyxy[:, 0]
        h  = result.xyxy[:, 3] - result.xyxy[:, 1]
        result.xywh = np.stack([cx, cy, w, h], axis=1)
        return result

    def __len__(self):
        return len(self.conf)


def _load_bytetrack_cfg(yaml_path="assets/bytetrack.yaml", frame_rate=25):
    """Đọc file config ByteTrack và trả về IterableSimpleNamespace."""
    try:
        from ultralytics.utils import IterableSimpleNamespace
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["frame_rate"] = frame_rate  # Thêm frame_rate vào config
        return IterableSimpleNamespace(**cfg)
    except Exception as e:
        logger.warning(f"Không đọc được bytetrack.yaml: {e}. Dùng config mặc định.")
        from ultralytics.utils import IterableSimpleNamespace
        return IterableSimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=0.25,
            track_low_thresh=0.05,
            new_track_thresh=0.25,
            track_buffer=30,
            match_thresh=0.8,
            fuse_score=True,
            frame_rate=frame_rate,
        )


class TrackingThread(QThread):
    """
    Nhận (frame, raw_dets) từ ProcessThread.
    raw_dets: numpy array (N, 6) = [x1, y1, x2, y2, conf, cls]
    Chạy BYTETracker để gán ID, vẽ Trajectory, rồi đẩy sang LogicThread.
    """
    fps_signal = pyqtSignal(float)

    def __init__(self, tracking_queue, logic_queue, frame_rate=25):
        super().__init__()
        self.tracking_queue = tracking_queue
        self.logic_queue = logic_queue
        self.running = True

        # Khởi tạo BYTETracker — frame_rate nằm trong cfg
        cfg = _load_bytetrack_cfg(frame_rate=frame_rate)
        self.tracker = BYTETracker(cfg)

        self.trajectories = {}          # track_id -> list of (cx, cy)
        self.max_trajectory_length = 20

        self._track_fps = 0.0
        self._last_time = time.time()
        self._last_fps_emit = 0.0

    def run(self):
        logger.info("TrackingThread bắt đầu chạy (BYTETracker).")
        while self.running:
            try:
                frame, raw_dets = self.tracking_queue.get(timeout=1)

                # Tạo đối tượng giả lập Results để BYTETracker có thể xử lý
                # BYTETracker.update() cần: .xyxy, .conf, .cls
                if len(raw_dets) > 0:
                    det_obj = DetectionResult(
                        xyxy=raw_dets[:, :4],
                        conf=raw_dets[:, 4],
                        cls=raw_dets[:, 5],
                    )
                    # online_targets: numpy (M, 8) = [x1,y1,x2,y2, track_id, score, cls, idx]
                    online_targets = self.tracker.update(det_obj, frame)
                else:
                    online_targets = np.empty((0, 8), dtype=np.float32)

                # Xây dựng objects dict: {track_id: (centroid, (x1,y1,x2,y2))}
                objects = {}
                current_ids = set()

                # online_targets columns: [x1, y1, x2, y2, track_id, score, cls, idx]
                for row in online_targets:
                    x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                    track_id = int(row[4])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    objects[track_id] = ((cx, cy), (x1, y1, x2, y2))
                    current_ids.add(track_id)

                # Dọn dẹp trajectory của ID đã biến mất
                for oid in list(self.trajectories.keys()):
                    if oid not in current_ids:
                        del self.trajectories[oid]

                # Cập nhật và vẽ Trajectory
                for track_id, (centroid, _) in objects.items():
                    if track_id not in self.trajectories:
                        self.trajectories[track_id] = []
                    self.trajectories[track_id].append(centroid)
                    if len(self.trajectories[track_id]) > self.max_trajectory_length:
                        self.trajectories[track_id].pop(0)

                    # Vẽ điểm tâm
                    cv2.circle(frame, centroid, 4, (0, 255, 255), -1)

                    # Vẽ đường quỹ đạo (dày → mảnh dần)
                    pts = self.trajectories[track_id]
                    for i in range(1, len(pts)):
                        thickness = int(np.sqrt(self.max_trajectory_length / float(i + 1)) * 2.5)
                        cv2.line(frame, pts[i - 1], pts[i], (0, 255, 255), max(1, thickness))

                # Tính và emit FPS (mỗi 1 giây)
                now = time.time()
                diff = now - self._last_time
                if diff > 0:
                    self._track_fps = self._track_fps * 0.9 + (1.0 / diff) * 0.1
                self._last_time = now
                if now - self._last_fps_emit >= 1.0:
                    self.fps_signal.emit(self._track_fps)
                    self._last_fps_emit = now

                # Đẩy sang luồng Logic
                if self.logic_queue.empty():
                    self.logic_queue.put((frame, objects))

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Lỗi trong TrackingThread: {e}")

    def stop(self):
        self.running = False
        self.wait()
