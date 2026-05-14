import sys
import torch  
from PyQt5.QtWidgets import QApplication
from src.ui.main_window import MainWindow
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())