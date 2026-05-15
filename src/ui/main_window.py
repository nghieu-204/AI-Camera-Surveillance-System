import queue
import cv2
from datetime import datetime
from PyQt5.QtWidgets import (QMainWindow, QLabel, QVBoxLayout, QHBoxLayout,
                              QWidget, QPushButton, QGridLayout,
                              QListWidget, QListWidgetItem, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, QPoint, QRect

from src.threads.capture import CaptureThread
from src.threads.process import ProcessThread
from src.threads.tracking import TrackingThread
from src.threads.logic import LogicThread
from src.utils.logger import logger
from src.utils.config_loader import (
    get_camera_sources, get_cameras_cfg, get_detection_cfg,
    get_tracking_cfg, get_logic_cfg, get_ui_cfg
)

class VideoLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.roi_points = []
        self.drawing_mode = False
        self.is_drawing = False
        self.rect_start = None
        self.rect_end = None
        self.roi_callback = None    # Gọi khi vẽ xong ROI (rel_points) -> (rel_points)
        self.status_callback = None # Gọi để cập nhật trạng thái ROI label

    def mousePressEvent(self, event):
        if self.drawing_mode and event.button() == Qt.LeftButton:
            self.is_drawing = True
            self.rect_start = event.pos()
            self.rect_end = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing_mode and self.is_drawing:
            self.rect_end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drawing_mode and event.button() == Qt.LeftButton and self.is_drawing:
            self.is_drawing = False
            self.rect_end = event.pos()
            
            tl = QPoint(min(self.rect_start.x(), self.rect_end.x()), min(self.rect_start.y(), self.rect_end.y()))
            br = QPoint(max(self.rect_start.x(), self.rect_end.x()), max(self.rect_start.y(), self.rect_end.y()))
            tr = QPoint(br.x(), tl.y())
            bl = QPoint(tl.x(), br.y())
            
            self.roi_points = [tl, tr, br, bl]
            self.drawing_mode = False
            self.update()
            
            rel_points = [(p.x() / self.width(), p.y() / self.height()) for p in self.roi_points]
            if self.roi_callback:
                self.roi_callback(rel_points)
                logger.info("Rectangle ROI set.")
            if self.status_callback:
                self.status_callback(True)

    def set_roi_callback(self, callback):
        self.roi_callback = callback

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        pen = QPen(Qt.red, 2, Qt.SolidLine)
        painter.setPen(pen)

        # Draw while dragging
        if self.drawing_mode and self.is_drawing and self.rect_start and self.rect_end:
            rect = QRect(self.rect_start, self.rect_end)
            painter.drawRect(rect)


class CameraWidget(QWidget):
    def __init__(self, camera_id, source="assets/test.mp4", parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self.source = source
        self.is_playing = True
        self.alert_panel_callback = None  # callback để gửi event ra panel tổng
        
        main_layout = QVBoxLayout()
        
        # --- Top Buttons ---
        btn_layout = QHBoxLayout()
        self.lbl_title = QLabel(f"Camera {self.camera_id}")
        self.lbl_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        
        self.btn_play_pause = QPushButton("Pause")
        self.btn_play_pause.clicked.connect(self.toggle_pause)
        
        self.btn_draw_roi = QPushButton("Draw ROI")
        self.btn_draw_roi.clicked.connect(self.enable_drawing)
        
        self.btn_clear_roi = QPushButton("Clear ROI")
        self.btn_clear_roi.clicked.connect(self.clear_roi)
        
        btn_layout.addWidget(self.lbl_title)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_play_pause)
        btn_layout.addWidget(self.btn_draw_roi)
        btn_layout.addWidget(self.btn_clear_roi)
        
        # --- Video Label ---
        self.video_label = VideoLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border: 1px solid gray;")
        self.video_label.setMinimumSize(400, 300)
        self.video_label.status_callback = self.update_roi_status
        
        # --- Bottom Status ---
        status_layout = QHBoxLayout()
        self.lbl_status = QLabel("Status: Play")
        self.lbl_roi_status = QLabel("ROI: Inactive")
        self.lbl_cap_fps = QLabel("Cap: 0.0")
        self.lbl_proc_fps = QLabel("Proc: 0.0")
        self.lbl_track_fps = QLabel("Track: 0.0")
        self.lbl_latency = QLabel("Lat: 0ms")
        self.lbl_count = QLabel("People: 0")
        
        for lbl in [self.lbl_status, self.lbl_roi_status,
                    self.lbl_cap_fps, self.lbl_proc_fps,
                    self.lbl_track_fps, self.lbl_latency, self.lbl_count]:
            lbl.setStyleSheet("font-size: 12px; font-weight: bold;")
            status_layout.addWidget(lbl)
            
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(self.video_label)
        main_layout.addLayout(status_layout)
        self.setLayout(main_layout)
        
        # Init Queues (Pipeline: capture -> process -> tracking -> logic -> UI)
        self.frame_queue = queue.Queue(maxsize=2)
        self.tracking_queue = queue.Queue(maxsize=2)
        self.logic_queue = queue.Queue(maxsize=2)
        
        # Init Threads
        cam_cfg = get_cameras_cfg()
        det = get_detection_cfg()
        trk = get_tracking_cfg()
        lgc = get_logic_cfg()
        ui  = get_ui_cfg()

        self.capture_thread  = CaptureThread(
            source=self.source,
            frame_queue=self.frame_queue,
            reconnect_delay=cam_cfg["reconnect_delay"]
        )
        self.process_thread  = ProcessThread(
            frame_queue=self.frame_queue,
            tracking_queue=self.tracking_queue,
            model_path=det["model_path"],
            conf_threshold=det["conf_threshold"],
            iou_threshold=det["iou_threshold"],
            imgsz=det["imgsz"],
        )
        self.tracking_thread = TrackingThread(
            tracking_queue=self.tracking_queue,
            logic_queue=self.logic_queue,
            frame_rate=trk["frame_rate"],
        )
        self.tracking_thread.max_trajectory_length = trk["max_trajectory_length"]
        self.logic_thread    = LogicThread(
            logic_queue=self.logic_queue,
            target_ui_fps=ui["target_ui_fps"],
        )
        self.logic_thread.dwell_threshold = lgc["dwell_threshold_seconds"]
        self.logic_thread.crowd_threshold = lgc["crowd_threshold"]
        self.logic_thread.camera_id = self.camera_id
        
        # Wire ROI callback: cập nhật cả 2 thread khi người dùng vẽ ROI
        self.video_label.set_roi_callback(self.on_roi_drawn)
        self.video_label.status_callback = self.update_roi_status
        
        # Connect signals
        self.capture_thread.fps_signal.connect(self.update_cap_fps)
        self.process_thread.fps_signal.connect(self.update_proc_fps)
        self.tracking_thread.fps_signal.connect(self.update_track_fps)
        self.logic_thread.result_signal.connect(self.update_ui)
    def start(self):
        self.capture_thread.start()
        self.process_thread.start()
        self.tracking_thread.start()
        self.logic_thread.start()
        
    def stop(self):
        self.capture_thread.stop()
        self.process_thread.stop()
        self.tracking_thread.stop()
        self.logic_thread.stop()

    def toggle_pause(self):
        paused = self.capture_thread.toggle_pause()
        if paused:
            self.btn_play_pause.setText("Play")
            self.lbl_status.setText("Status: Stop")
            self.lbl_status.setStyleSheet("font-size: 14px; font-weight: bold; color: red;")
        else:
            self.btn_play_pause.setText("Pause")
            self.lbl_status.setText("Status: Play")
            self.lbl_status.setStyleSheet("font-size: 14px; font-weight: bold; color: green;")

    def enable_drawing(self):
        self.video_label.drawing_mode = True
        self.video_label.roi_points = []
        self.video_label.rect_start = None
        self.video_label.rect_end = None
        self.lbl_roi_status.setText("ROI: Drawing...")

    def on_roi_drawn(self, rel_points):
        """Gọi khi người dùng vẽ xong ROI - cập nhật cả 2 thread."""
        self.logic_thread.set_roi(rel_points)
        self.process_thread.set_roi(rel_points)

    def clear_roi(self):
        self.video_label.roi_points = []
        self.video_label.drawing_mode = False
        self.video_label.update()
        self.logic_thread.set_roi([])
        self.process_thread.set_roi([])
        self.update_roi_status(False)

    def update_roi_status(self, is_active):
        self.process_thread.set_roi_active(is_active)
        if is_active:
            self.lbl_roi_status.setText("ROI: Active")
            self.lbl_roi_status.setStyleSheet("font-size: 14px; font-weight: bold; color: blue;")
        else:
            self.lbl_roi_status.setText("ROI: Inactive")
            self.lbl_roi_status.setStyleSheet("font-size: 14px; font-weight: bold; color: black;")

    def update_cap_fps(self, fps):
        self.lbl_cap_fps.setText(f"Cap: {fps:.1f}")

    def update_proc_fps(self, fps):
        self.lbl_proc_fps.setText(f"Proc: {fps:.1f}")

    def update_track_fps(self, fps):
        self.lbl_track_fps.setText(f"Track: {fps:.1f}")

    def update_ui(self, frame, count, warning, logic_fps, alert_events, latency):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_img).scaled(self.video_label.width(), self.video_label.height(), Qt.IgnoreAspectRatio)
        self.video_label.setPixmap(pixmap)
        
        self.lbl_count.setText(f"People: {count}")
        self.lbl_latency.setText(f"Lat: {latency*1000:.0f}ms")
        
        if warning:
            self.lbl_count.setStyleSheet("font-size: 14px; font-weight: bold; color: red;")
        else:
            self.lbl_count.setStyleSheet("font-size: 14px; font-weight: bold; color: black;")
        
        # Gửi event cảnh báo về panel tổng
        if alert_events and self.alert_panel_callback:
            for event in alert_events:
                self.alert_panel_callback(event)

    def set_alert_callback(self, callback):
        self.alert_panel_callback = callback


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        ui = get_ui_cfg()
        self.setWindowTitle(ui["window_title"])
        self.setGeometry(50, 50, ui["window_width"], ui["window_height"])
        
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        # Root layout: camera grid (left) + alert panel (right)
        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(6)
        
        # --- Left: 2x2 Camera Grid ---
        grid_layout = QGridLayout()
        grid_layout.setSpacing(4)
        
        self.cameras = []
        sources = get_camera_sources()
        for i in range(4):
            source = sources[i] if i < len(sources) else "assets/test.mp4"
            cam = CameraWidget(camera_id=i+1, source=source)
            cam.set_alert_callback(self.add_alert)
            self.cameras.append(cam)
            row = i // 2
            col = i % 2
            grid_layout.addWidget(cam, row, col)
            cam.start()
        
        cam_container = QWidget()
        cam_container.setLayout(grid_layout)
        root_layout.addWidget(cam_container, stretch=4)
        
        # --- Right: Alert Panel ---
        alert_panel = QWidget()
        alert_panel.setMinimumWidth(260)
        alert_panel.setMaximumWidth(320)
        alert_panel.setStyleSheet("background-color: #1a1a2e; border-radius: 8px;")
        alert_layout = QVBoxLayout(alert_panel)
        alert_layout.setContentsMargins(8, 10, 8, 10)
        alert_layout.setSpacing(6)
        
        lbl_alert_title = QLabel("CẢNH BÁO REALTIME")
        lbl_alert_title.setStyleSheet(
            "color: #e94560; font-size: 14px; font-weight: bold;"
            "padding: 6px; background: #16213e; border-radius: 6px;"
        )
        lbl_alert_title.setAlignment(Qt.AlignCenter)
        alert_layout.addWidget(lbl_alert_title)
        
        self.alert_list = QListWidget()
        self.alert_list.setWordWrap(True)
        self.alert_list.setSpacing(3)
        self.alert_list.setStyleSheet("""
            QListWidget {
                background-color: #16213e;
                color: #eaeaea;
                font-size: 12px;
                border: none;
                border-radius: 6px;
            }
            QListWidget::item {
                padding: 6px 8px;
                border-bottom: 1px solid #0f3460;
            }
            QListWidget::item:selected {
                background-color: #0f3460;
            }
        """)
        alert_layout.addWidget(self.alert_list, stretch=1)
        
        btn_clear = QPushButton("\U0001f5d1  Xóa cảnh báo")
        btn_clear.setStyleSheet(
            "background-color: #0f3460; color: white; font-size: 12px;"
            "padding: 7px; border-radius: 6px; border: none;"
        )
        btn_clear.clicked.connect(self.alert_list.clear)
        alert_layout.addWidget(btn_clear)
        
        root_layout.addWidget(alert_panel, stretch=1)

    def add_alert(self, message: str):
        """Thêm một event cảnh báo mới vào đầu danh sách."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        item = QListWidgetItem(f"[{timestamp}]\n{message}")
        
        if "ĐÁM ĐÔNG" in message:
            item.setForeground(QColor("#ff6b6b"))
        else:
            item.setForeground(QColor("#ffd166"))
        
        self.alert_list.insertItem(0, item)
        
        # Giới hạn tối đa 100 sự kiện
        if self.alert_list.count() > 100:
            self.alert_list.takeItem(self.alert_list.count() - 1)

    def closeEvent(self, event):
        logger.info("Đang đóng ứng dụng...")
        for cam in self.cameras:
            cam.stop()
        event.accept()