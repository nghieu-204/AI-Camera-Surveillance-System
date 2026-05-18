"""
config_loader.py — Đọc và cung cấp cấu hình từ config.yaml ra toàn bộ ứng dụng.
Dùng Singleton pattern: chỉ đọc file 1 lần, các module khác import trực tiếp.
"""
import yaml
import os
from src.utils.logger import logger

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")

# Giá trị mặc định — dùng khi config.yaml bị thiếu hoặc lỗi
_DEFAULTS = {
    "cameras": {
        "sources": ["assets/test.mp4"] * 4,
        "reconnect_delay": 2,
    },
    "detection": {
        "model_path": "yolov8n.pt",
        "imgsz": 640,
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
        "device": "auto",
    },
    "tracking": {
        "bytetrack_config": "assets/bytetrack.yaml",
        "frame_rate": 25,
        "max_trajectory_length": 20,
    },
    "logic": {
        "dwell_threshold_seconds": 3.0,
        "crowd_threshold": 5,
    },
    "ui": {
        "target_ui_fps": 15,
        "max_alerts": 100,
        "window_width": 1280,
        "window_height": 720,
        "window_title": "AI Camera Surveillance System",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Gộp override vào base, ưu tiên giá trị trong override."""
    result = base.copy()
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _load() -> dict:
    path = os.path.normpath(_CONFIG_PATH)
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        cfg = _deep_merge(_DEFAULTS, raw)
        logger.info(f"Đã tải cấu hình từ: {path}")
        return cfg
    except FileNotFoundError:
        logger.warning(f"Không tìm thấy {path}. Dùng cấu hình mặc định.")
        return _DEFAULTS.copy()
    except Exception as e:
        logger.error(f"Lỗi đọc config.yaml: {e}. Dùng cấu hình mặc định.")
        return _DEFAULTS.copy()


# Singleton — load 1 lần khi module được import
cfg = _load()

# --- Shortcut accessors ---
def get_cameras_cfg() -> dict:
    return cfg["cameras"]

def get_camera_sources() -> list[str]:
    return cfg["cameras"]["sources"]

def get_detection_cfg() -> dict:
    return cfg["detection"]

def get_tracking_cfg() -> dict:
    return cfg["tracking"]

def get_logic_cfg() -> dict:
    return cfg["logic"]

def get_ui_cfg() -> dict:
    return cfg["ui"]
