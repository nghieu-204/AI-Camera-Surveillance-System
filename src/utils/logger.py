import logging
import os
from datetime import datetime

def setup_logger():
    # Tạo thư mục logs nếu chưa tồn tại
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Tên file log dựa trên ngày
    log_filename = datetime.now().strftime("logs/system_%Y-%m-%d.log")

    # Cấu hình logger
    logger = logging.getLogger("AI_Camera_Logger")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # Ghi ra file
        file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)

        # Ghi ra console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Định dạng log
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

logger = setup_logger()
