import cv2
import time
from PyQt5.QtCore import QThread, pyqtSignal
from src.utils.logger import logger

class CaptureThread(QThread):
    # Tín hiệu gửi fps ra UI
    fps_signal = pyqtSignal(float) 
    
    def __init__(self, source, frame_queue, reconnect_delay=2):
        super().__init__()
        self.source = source
        self.frame_queue = frame_queue
        self.reconnect_delay = reconnect_delay
        self.running = True
        self.paused = False
        self.last_time = time.time()
        self.cap_fps = 0.0
        self._last_fps_emit = 0.0  # Throttle fps_signal (chỉ emit mỗi 1 giây)

    def run(self):
        logger.info(f"Bắt đầu đọc video từ: {self.source}")
        cap = cv2.VideoCapture(self.source)
        
        # Lấy FPS của video
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps != fps: # Handle 0 or NaN
            fps = 30.0
        frame_delay = 1.0 / fps

        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue
                
            start_time = time.time()
            ret, frame = cap.read()
            if ret:
                current_time = time.time()
                # Throttle fps_signal: chỉ emit mỗi 1 giây
                if current_time - self._last_fps_emit >= 1.0:
                    self.fps_signal.emit(self.cap_fps)
                    self._last_fps_emit = current_time

                # Đẩy frame xuống pipeline: dùng backpressure tự nhiên của queue
                if self.frame_queue.empty():
                    self.frame_queue.put(frame)
                    time_diff = current_time - self.last_time
                    if time_diff > 0:
                        self.cap_fps = (self.cap_fps * 0.9) + ((1.0 / time_diff) * 0.1)
                    self.last_time = current_time
            else:
                logger.warning(f"Mất tín hiệu hoặc hết video từ: {self.source}. Đang thử kết nối lại...")
                cap.release()
                time.sleep(self.reconnect_delay) # Reconnect delay
                cap = cv2.VideoCapture(self.source)
                continue
                
            # Đợi để video chạy đúng tốc độ
            elapsed = time.time() - start_time
            sleep_time = frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        cap.release()
        logger.info("CaptureThread đã dừng.")

    def stop(self):
        self.running = False
        self.wait()

    def toggle_pause(self):
        self.paused = not self.paused
        return self.paused
