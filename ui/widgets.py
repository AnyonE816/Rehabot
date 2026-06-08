# ui/widgets.py
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class VideoDisplay(QLabel):
    """自定义视频显示控件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Waiting for video...")

    def set_image(self, image: QImage):
        """设置要显示的图像帧"""
        self.setPixmap(QPixmap.fromImage(image))