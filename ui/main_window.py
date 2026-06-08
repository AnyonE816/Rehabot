# ui/main_window.py
import sys
import os
import csv
import time
import datetime
import threading
import subprocess
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QHBoxLayout,
                             QGraphicsDropShadowEffect, QTableView, QTextBrowser, QGroupBox, 
                             QLineEdit, QPushButton, QFileDialog, QRadioButton, QLCDNumber, 
                             QMessageBox, QScrollArea, QProgressBar, QAbstractItemView)
from PyQt5.QtGui import QImage, QPixmap, QColor, QFont, QStandardItemModel, QStandardItem
from PyQt5.QtCore import QTimer, Qt

# 导入核心模块
from core.pose_detector import PoseDetector
from ui.widgets import VideoDisplay
from config import *

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pose Detection UI")
        self.setGeometry(210, 160, 2200, 1300)
        main_layout = QHBoxLayout()
        main_layout.setSpacing(30)
        video_group_box = QGroupBox()
        video_group_box.setStyleSheet("background: transparent; border: 5px dashed pink;")
        video_layout = QVBoxLayout()
        video_layout.setSpacing(5)
        video_horizontal_layout = QHBoxLayout()
        video_horizontal_layout.setSpacing(10)
        video1_group_box = QGroupBox()
        video1_group_box.setStyleSheet("background: transparent; border: none;")
        video1_layout = QVBoxLayout()
        video1_layout.setSpacing(30)
        self.video1_display = VideoDisplay()
        self.video1_display.setFixedSize(720, 600)
        self.video1_display.setStyleSheet("border: 5px dashed lightblue;")
        self.add_shadow_effect(self.video1_display, QColor(0, 0, 255, 150))
        self.video1_label = QLabel("Demonstration Video")
        self.video1_label.setStyleSheet("color: black; font-size: 36px; font-weight: bold; font-family: 'Times New Roman'; border: none;")
        self.video1_label.setAlignment(Qt.AlignCenter)
        video1_layout.addWidget(self.video1_label)
        video1_layout.addWidget(self.video1_display)
        video1_group_box.setLayout(video1_layout)
        video2_group_box = QGroupBox()
        video2_group_box.setStyleSheet("background: transparent; border: none;")
        video2_layout = QVBoxLayout()
        video2_layout.setSpacing(30)
        self.video2_display = VideoDisplay()
        self.video2_display.setFixedSize(720, 600)
        self.add_shadow_effect(self.video2_display, QColor(255, 0, 0, 150))
        self.video2_display.setStyleSheet("border: 5px dashed pink;")
        self.video2_label = QLabel("User Screen")
        self.video2_label.setStyleSheet("color: black; font-size: 36px; font-weight: bold; font-family: 'Times New Roman'; border: none;")
        self.video2_label.setAlignment(Qt.AlignCenter)
        video2_layout.addWidget(self.video2_label)
        video2_layout.addWidget(self.video2_display)
        video2_group_box.setLayout(video2_layout)
        video_horizontal_layout.addWidget(video1_group_box)
        video_horizontal_layout.addWidget(video2_group_box)
        video_layout.addLayout(video_horizontal_layout)
        videodata_group_box = QGroupBox()
        videodata_group_box.setStyleSheet("background: transparent; border: none;")
        videodata_layout = QVBoxLayout()
        videodata_layout.setSpacing(20)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet(
            "QProgressBar { border: 5px dashed darkgreen; border-radius: 10px; font-size: 26px; text-align: center; padding: 10px; }"
            "QProgressBar::chunk { background-color: #05B8CC; width: 20px; }")
        self.progress_bar.setFixedWidth(1480)
        progress_bar_layout = QHBoxLayout()
        progress_bar_layout.addStretch(1)
        progress_bar_layout.addWidget(self.progress_bar)
        progress_bar_layout.addStretch(1)
        videodata_layout.addLayout(progress_bar_layout)
        video_layout.addWidget(videodata_group_box)
        videodata_group_box.setLayout(videodata_layout)
        video_group_box.setLayout(video_layout)
        self.text_browser_videodata = QTextBrowser()
        self.text_browser_videodata.setPlainText("Action guide statements are all here!")
        self.text_browser_videodata.setFont(QFont("Times New Roman", 12))
        self.text_browser_videodata.setMinimumHeight(100)
        self.text_browser_videodata.setStyleSheet("background: transparent; border: 5px dashed pink;")
        self.info_messages = []
        self.table_view = QTableView()
        self.table_view.setMinimumHeight(200)
        self.table_model = QStandardItemModel()
        self.table_model.setHorizontalHeaderLabels([
            "Start_Time", "End_Time", "Demonstration_Video_Path", "Playback_Save_Path", "Report_Save_Path",
            "Number_of_Training_Videos", "Total_Number_of_Errors", "count_angle_left", "count_offset_left",
            "count_angle_right", "count_offset_right", "Guidance_Recommendations", "Overall_Score"])
        self.table_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table_view.setModel(self.table_model)
        self.table_view.setStyleSheet("QTableView {border: 5px dashed pink; background: transparent}")
        videodata_layout.addWidget(self.text_browser_videodata)
        videodata_layout.addWidget(self.table_view)

        label_style = """color: black; font-size: 36px; font-weight: bold; font-family: 'Times New Roman'; border: none;"""
        self.intro_label = QLabel("Introduction")
        self.file_import_label = QLabel("Video Importing")
        self.playback_label = QLabel("Playback Settings")
        self.report_label = QLabel("Report Settings")
        self.detection_control_label = QLabel("Detection Control")
        self.intro_label.setStyleSheet(label_style)
        self.file_import_label.setStyleSheet(label_style)
        self.playback_label.setStyleSheet(label_style)
        self.report_label.setStyleSheet(label_style)
        self.detection_control_label.setStyleSheet(label_style)
        self.intro_label.setAlignment(Qt.AlignCenter)
        self.file_import_label.setAlignment(Qt.AlignCenter)
        self.playback_label.setAlignment(Qt.AlignCenter)
        self.report_label.setAlignment(Qt.AlignCenter)
        self.detection_control_label.setAlignment(Qt.AlignCenter)
        self.text_browser = QTextBrowser()
        self.text_browser.setPlainText("Welcome to Rehabot!\n\nA Real-Time Pose Detection and Feedback System for Upper Limb Rehabilitation")
        font = QFont("Times New Roman", 14)
        self.text_browser.setFont(font)
        self.text_browser.setMinimumWidth(400)
        self.text_browser.setMaximumHeight(154)
        self.text_browser.setStyleSheet("background: transparent; border: 5px dashed pink;")

        self.group_box_fileimport = QGroupBox()
        self.group_box_fileimport.setStyleSheet("background: transparent; border: 5px dashed pink;")
        self.group_box_fileimport.setMinimumWidth(400)
        self.group_box_fileimport.setMaximumHeight(150)
        group_box_layout = QVBoxLayout()
        self.key_edit1 = QLineEdit()
        self.key_edit1.setPlaceholderText("Please Select Video...")
        self.key_edit1.setFont(QFont("Times New Roman", 12))
        self.key_edit1.setMinimumHeight(50)
        self.push_button1 = QPushButton("🎬")
        self.push_button1.setFont(QFont("Times New Roman", 12))
        self.push_button1.setFixedSize(50, 50)
        self.push_button1.clicked.connect(self.select_video_file)
        row1_layout = QHBoxLayout()
        row1_layout.addWidget(self.key_edit1)
        row1_layout.addWidget(self.push_button1)
        self.key_edit2 = QLineEdit()
        self.key_edit2.setPlaceholderText("Please Select Folder...")
        self.key_edit2.setFont(QFont("Times New Roman", 12))
        self.key_edit2.setMinimumHeight(50)
        self.push_button2 = QPushButton("📂")
        self.push_button2.setFont(QFont("Times New Roman", 12))
        self.push_button2.setFixedSize(50, 50)
        self.push_button2.clicked.connect(self.select_folder_video)
        row2_layout = QHBoxLayout()
        row2_layout.addWidget(self.key_edit2)
        row2_layout.addWidget(self.push_button2)
        group_box_layout.addLayout(row1_layout)
        group_box_layout.addLayout(row2_layout)
        self.group_box_fileimport.setLayout(group_box_layout)
        self.playback_group_box = QGroupBox()
        self.playback_group_box.setStyleSheet("background: transparent; border: 5px dashed pink;")
        self.playback_group_box.setMinimumWidth(400)
        self.playback_group_box.setMaximumHeight(220)
        playback_group_box_layout = QVBoxLayout()
        self.enable_playback_radio = QRadioButton("Enable Playback")
        self.enable_playback_radio.setFixedHeight(50)
        self.enable_playback_radio.setFont(QFont("Times New Roman", 12))
        self.enable_playback_radio.toggled.connect(self.on_radio_button_toggled)
        self.save_playback_button = QPushButton("Playback Library")
        self.save_playback_button.setFont(QFont("Times New Roman", 12))
        self.save_playback_button.setFixedSize(200, 50)
        self.save_playback_button.clicked.connect(self.Playback_Library)
        playback_up_layout = QHBoxLayout()
        playback_up_layout.addWidget(self.enable_playback_radio)
        playback_up_layout.addWidget(self.save_playback_button)
        playback_group_box_layout.addLayout(playback_up_layout)
        self.playback_edit = QLineEdit()
        self.playback_edit.textChanged.connect(self.on_playback_text_changed)
        self.playback_edit.setPlaceholderText("Please Select Playback Save Path...")
        self.playback_edit.setFont(QFont("Times New Roman", 12))
        self.playback_edit.setMinimumHeight(50)
        self.playback_button = QPushButton("📼")
        self.playback_button.setFont(QFont("Times New Roman", 12))
        self.playback_button.setFixedSize(50, 50)
        self.playback_button.clicked.connect(self.select_folder_playback)
        playback_row_layout = QHBoxLayout()
        playback_row_layout.addWidget(self.playback_edit)
        playback_row_layout.addWidget(self.playback_button)
        playback_group_box_layout.addLayout(playback_row_layout)
        self.playback_group_box.setLayout(playback_group_box_layout)
        self.playback_time_edit = QLineEdit("Playback Time")
        self.playback_time_edit.setFont(QFont("Times New Roman", 12))
        self.playback_time_edit.setMinimumHeight(50)
        self.playback_time_edit.setReadOnly(True)
        self.playback_lcd = QLCDNumber()
        self.playback_lcd.setDigitCount(8)
        self.playback_lcd.setSegmentStyle(QLCDNumber.Flat)
        self.playback_lcd.setMinimumHeight(50)
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.update_playback_lcd)
        self.playback_time = 0
        playback_time_row_layout = QHBoxLayout()
        playback_time_row_layout.addWidget(self.playback_time_edit)
        playback_time_row_layout.addWidget(self.playback_lcd)
        playback_group_box_layout.addLayout(playback_time_row_layout)
        self.playback_group_box.setLayout(playback_group_box_layout)


        self.report_group_box = QGroupBox()
        self.report_group_box.setStyleSheet("background: transparent; border: 5px dashed pink;")
        self.report_group_box.setMinimumWidth(400)
        self.report_group_box.setMaximumHeight(150)
        report_group_box_layout = QVBoxLayout()
        self.enable_report_radio = QRadioButton("Enable Report")
        self.enable_report_radio.setFixedHeight(50)
        self.enable_report_radio.setFont(QFont("Times New Roman", 12))
        self.enable_report_radio.toggled.connect(self.on_report_radio_button_toggled)
        self.save_report_button = QPushButton("Save Report")
        self.save_report_button.setFont(QFont("Times New Roman", 12))
        self.save_report_button.setFixedSize(200, 50)
        self.save_report_button.clicked.connect(self.save_report)
        report_up_layout = QHBoxLayout()
        report_up_layout.addWidget(self.enable_report_radio)
        report_up_layout.addWidget(self.save_report_button)
        report_group_box_layout.addLayout(report_up_layout)
        self.report_edit = QLineEdit()
        self.report_edit.textChanged.connect(self.on_report_text_changed)
        self.report_edit.setPlaceholderText("Please Select Report Save Path...")
        self.report_edit.setFont(QFont("Times New Roman", 12))
        self.report_edit.setMinimumHeight(50)
        self.report_button = QPushButton("📋")
        self.report_button.setFont(QFont("Times New Roman", 12))
        self.report_button.setFixedSize(50, 50)
        self.report_button.clicked.connect(self.select_folder_report)

        report_row_layout = QHBoxLayout()
        report_row_layout.addWidget(self.report_edit)
        report_row_layout.addWidget(self.report_button)
        report_group_box_layout.addLayout(report_row_layout)

        self.report_group_box.setLayout(report_group_box_layout)
        self.control_group_box = QGroupBox()
        self.control_group_box.setStyleSheet("background: transparent; border: 5px dashed pink;")
        self.control_group_box.setMinimumWidth(400)
        self.control_group_box.setMaximumHeight(250)
        control_group_box_layout = QVBoxLayout()
        self.control_text_browser = QTextBrowser()
        self.control_text_browser.setPlainText("Please complete the above settings\nbefore starting the Detection")
        self.control_text_browser.setFont(QFont("Times New Roman", 12))
        self.control_text_browser.setMinimumHeight(40)
        self.control_text_browser.setMaximumHeight(74)
        control_group_box_layout.addWidget(self.control_text_browser)
        start_stop_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.setFont(QFont("Times New Roman", 12))
        self.start_button.setFixedSize(100, 50)
        self.start_button.clicked.connect(self.start_detection)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setFont(QFont("Times New Roman", 12))
        self.stop_button.setFixedSize(100, 50)
        self.stop_button.clicked.connect(self.stop_detection)
        start_stop_layout.addWidget(self.start_button)
        start_stop_layout.addWidget(self.stop_button)
        control_group_box_layout.addLayout(start_stop_layout)
        reset_help_layout = QHBoxLayout()
        self.reset_button = QPushButton("Reset")
        self.reset_button.setFont(QFont("Times New Roman", 12))
        self.reset_button.setFixedSize(100, 50)
        self.reset_button.clicked.connect(self.reset_line_edits)
        self.help_button = QPushButton("Help")
        self.help_button.setFont(QFont("Times New Roman", 12))
        self.help_button.setFixedSize(100, 50)
        self.help_button.clicked.connect(self.show_help_message)
        reset_help_layout.addWidget(self.reset_button)
        reset_help_layout.addWidget(self.help_button)
        control_group_box_layout.addLayout(reset_help_layout)
        self.control_group_box.setLayout(control_group_box_layout)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.intro_label)
        right_layout.addWidget(self.text_browser)
        right_layout.addWidget(self.file_import_label)
        right_layout.addWidget(self.group_box_fileimport)
        right_layout.addWidget(self.playback_label)
        right_layout.addWidget(self.playback_group_box)
        right_layout.addWidget(self.report_label)
        right_layout.addWidget(self.report_group_box)
        right_layout.addWidget(self.detection_control_label)
        right_layout.addWidget(self.control_group_box)
        right_scroll_area = QScrollArea()
        right_scroll_area.setWidgetResizable(True)
        right_scroll_widget = QWidget()
        right_scroll_widget.setLayout(right_layout)
        right_scroll_area.setWidget(right_scroll_widget)

        main_layout.addWidget(video_group_box, stretch=2)
        main_layout.addWidget(right_scroll_area, stretch=1)
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self.is_running = False
        self.cam1 = None
        self.cam2 = None
        self.total_frames_video1 = 0
        self.current_frame_video1 = 0
        self.detector1 = PoseDetector()
        self.detector2 = PoseDetector()
        self.angle_list = []
        self.start_time = None
        self.end_time = None
        self.demo_video_path = ""
        self.total_errors = 0
        self.count_angle_left = 0
        self.count_offset_left = 0
        self.count_angle_right = 0
        self.count_offset_right = 0
        self.recording = False
        self.video_writer = None
        self.recording_fps = 15
        self.recording_thread = None
        self.thread1 = None
        self.thread2 = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(30)
    def add_shadow_effect(self, widget, color):
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(color)
        shadow.setOffset(5, 5)
        widget.setGraphicsEffect(shadow)
    def dummy_callback(self):
        pass
    def select_video_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4)")
        if file_name:
            self.key_edit1.setText(file_name)
            self.cam1 = cv2.VideoCapture(file_name)
            self.total_frames_video1 = int(self.cam1.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame_video1 = 0
            self.key_edit2.clear()
            self.thread1 = threading.Thread(target=self.process_video, args=(self.cam1, self.detector1, "Video1", self.dummy_callback))
            self.thread1.start()
    def select_folder_video(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.key_edit2.setText(folder_path)
            self.video_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.mp4')]
            self.key_edit1.clear()
            self.current_video_index = 0
    def select_folder_playback(self):
        folder_path = QFileDialog.getExistingDirectory(self.playback_group_box, "Select Folder")
        if folder_path:
            self.playback_edit.setText(folder_path)
    def select_folder_report(self):
        folder_path = QFileDialog.getExistingDirectory(self.report_group_box, "Select Folder")
        if folder_path:
            self.report_edit.setText(folder_path)
    def on_report_text_changed(self):
        if not self.report_edit.text().strip():
            self.enable_report_radio.setChecked(False)
    def on_playback_text_changed(self):
        if not self.playback_edit.text().strip():
            self.enable_playback_radio.setChecked(False)
    def Playback_Library(self):
        try:
            path = self.playback_edit.text().strip()
            if not path or path == self.playback_edit.placeholderText():
                QMessageBox.warning(self, "路径未选择", "请先选择回放保存路径！")
                return
            if not os.path.exists(path):
                QMessageBox.critical(self, "路径错误", "指定路径不存在！")
                return
            if sys.platform == 'win32':
                os.startfile(path)
            elif sys.platform == 'darwin':
                subprocess.call(['open', path])
            else:
                subprocess.call(['xdg-open', path])
        except Exception as e:
            QMessageBox.critical(self, "打开失败", f"无法打开回放库：{str(e)}")

    def on_radio_button_toggled(self, checked):
        if checked:
            if self.playback_edit.text() == "" or self.playback_edit.text() == self.playback_edit.placeholderText():
                self.select_folder_playback()
    def on_report_radio_button_toggled(self, checked):
        if checked:
            if self.report_edit.text() == "" or self.report_edit.text() == self.report_edit.placeholderText():
                self.select_folder_report()
    def play_next_video(self):
        if self.current_video_index < len(self.video_files):
            video_file = self.video_files[self.current_video_index]
            self.key_edit1.setText(video_file)
            self.cam1 = cv2.VideoCapture(video_file)
            self.total_frames_video1 = int(self.cam1.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame_video1 = 0
            self.progress_bar.setValue(0)  # 重置进度条
            self.thread1 = threading.Thread(
                target=lambda: self.process_video(self.cam1, self.detector1, "Video1", self.play_next_video_callback))
            self.thread1.start()
            self.current_video_index += 1
        else:
            self.current_video_index = 0
            self.cam1 = None
            self.video1_display.setText("Waiting for video...")
    def play_next_video_callback(self):
        self.play_next_video()
    def start_detection(self):
        if not self.is_running:
            self.start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            self.playback_time = 0
            self.playback_lcd.display(self.playback_time)
            if self.enable_playback_radio.isChecked():
                self.playback_timer.start(1000)

            self.end_time = "None"
            self.total_errors = 0
            self.count_angle_left = 0
            self.count_offset_left = 0
            self.count_angle_right = 0
            self.count_offset_right = 0
            self.is_running = True
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.cam2 = cv2.VideoCapture(0)
            if self.enable_playback_radio.isChecked():
                save_dir = self.playback_edit.text()
                if save_dir in ("", self.playback_edit.placeholderText()):
                    QMessageBox.warning(self, "Warning", "Please select playback save path first!")
                    return
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                self.recording_path = os.path.join(save_dir, f"screen_record_{timestamp}.mp4")
                os.makedirs(save_dir, exist_ok=True)
                self.recording = True
                self.recording_thread = threading.Thread(target=self.screen_recording)
                self.recording_thread.start()

            # ======================================================================
            self.thread2 = threading.Thread(target=self.process_video,
                                            args=(self.cam2, self.detector2, "Video2", self.dummy_callback))
            # ======================================================================

            self.thread2.start()
            if self.cam1 is not None:
                self.demo_video_path = self.key_edit1.text()
                self.thread1 = threading.Thread(target=self.process_video,
                                                args=(self.cam1, self.detector1, "Video1", self.dummy_callback))
                self.thread1.start()
            elif hasattr(self, 'video_files') and self.video_files:
                self.demo_video_path = self.key_edit2.text()
                self.play_next_video()
    def stop_detection(self):
        if self.is_running:
            self.end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.playback_timer.stop()

            # ===== 新增录屏停止逻辑 =========================================
            # 安全停止录制
            if self.recording:
                self.recording = False
                if self.recording_thread and self.recording_thread.is_alive():
                    self.recording_thread.join(timeout=1)

            # 安全释放资源
            if hasattr(self, 'video_writer'):
                try:
                    if self.video_writer.isOpened():
                        self.video_writer.release()
                except:
                    pass
                finally:
                    del self.video_writer
            # =================================================================
            if self.enable_report_radio.isChecked():
                self.update_table_data()
            else:
                self.table_data_off()
            self.is_running = False
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.total_frames_video1 = 0
            self.current_frame_video1 = 0
            self.progress_bar.setValue(0)
            if self.thread1:
                self.thread1.join()
                self.thread1 = None
            if self.thread2:
                self.thread2.join()
                self.thread2 = None
            if self.cam1:
                self.cam1.release()
                self.cam1 = None
            if self.cam2:
                self.cam2.release()
                self.cam2 = None
            self.video1_display.setText("Waiting for video...")
            self.video2_display.setText("Waiting for video...")
    def reset_line_edits(self):
        self.key_edit1.clear()
        self.key_edit2.clear()
        self.playback_edit.clear()
        self.report_edit.clear()
        self.progress_bar.setValue(0)
        self.enable_playback_radio.setChecked(False)
        self.enable_report_radio.setChecked(False)
    def show_help_message(self):
        help_message_box = QMessageBox(self)
        help_message_box.setWindowTitle("Help")
        help_message_box.setText("aaa")
        help_message_box.setIcon(QMessageBox.Information)
        help_message_box.setModal(True)
        help_message_box.exec_()
    def save_report(self):
        """保存报表为CSV文件"""
        try:
            if self.table_model.rowCount() == 0:
                raise ValueError("No training data to save")
            if not self.enable_report_radio.isChecked():
                raise ValueError("Report saving is not enabled")

            report_dir = self.report_edit.text()
            if report_dir in ("", self.report_edit.placeholderText()):
                raise ValueError("Invalid report path")
            headers = [self.table_model.horizontalHeaderItem(i).text()
                       for i in range(self.table_model.columnCount())]

            rows = []
            for row_idx in range(self.table_model.rowCount()):
                row = [
                    self.table_model.item(row_idx, col_idx).text()
                    for col_idx in range(self.table_model.columnCount())
                ]
                rows.append(row)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pose_report_{timestamp}.csv"
            save_path = os.path.join(report_dir, filename)
            os.makedirs(report_dir, exist_ok=True)
            with open(save_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)

            QMessageBox.information(self, "Success", '"Report" saved successfully!')

        except Exception as e:
            error_msg = f'"Report" save failed!\nReason: {str(e)}'
            QMessageBox.critical(self, "Error", error_msg)
    def screen_recording(self):
        self.video_writer = None
        """屏幕录制线程函数"""
        while self.recording:
            try:
                pixmap = self.grab()
                if pixmap.isNull():
                    continue
                # img = pixmap.toImage()
                img = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
                width, height = img.width(), img.height()
                if width == 0 or height == 0:
                    continue
                img = img.convertToFormat(QImage.Format_RGB888)
                ptr = img.bits()
                ptr.setsize(img.byteCount())
                arr = np.array(ptr).reshape(height, width, 3)
                if self.video_writer is None:
                    if not os.path.exists(os.path.dirname(self.recording_path)):
                        os.makedirs(os.path.dirname(self.recording_path), exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.video_writer = cv2.VideoWriter(
                        self.recording_path,
                        fourcc,
                        self.recording_fps,
                        (width, height))
                    if not self.video_writer.isOpened():
                        raise RuntimeError("视频写入器初始化失败")
                self.video_writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
                time.sleep(1 / self.recording_fps)
            except Exception as e:
                print(f"录制错误: {str(e)}")
                self.recording = False
                break
    def process_video(self, cam, detector, window_name, callback):
        A_last_orange_time_left: float = 0.0
        A_last_orange_time_right: float = 0.0
        A_is_red_left: bool = False
        A_is_red_right: bool = False
        stime = 0
        while self.is_running:
            success, video = cam.read()
            if not success:
                break

            video = cv2.resize(video, (720, 600))
            if window_name == "Video1":
                self.current_frame_video1 = int(cam.get(cv2.CAP_PROP_POS_FRAMES))
            if window_name == "Video2":
                video = cv2.flip(video, 1)
            v_pro, lmlist = detector.Position(video)
            if len(lmlist) != 0:
                angle_left = detector.Angle(v_pro, 11, 13, 15, True, color=(255, 255, 255))
                percent_left = np.interp(angle_left, (0, 180), (100, 0))
                angle_right = detector.Angle(v_pro, 12, 14, 16, True, color=(255, 255, 255))
                percent_right = np.interp(angle_right, (0, 180), (100, 0))
                self.angle_list.append((angle_left, angle_right))
                offset_left = detector.X_Offset(13, 23)
                offset_right = detector.X_Offset(14, 24)
                if window_name == "Video2" and len(self.angle_list) >= 2:
                    # angle1_left, angle1_right = self.angle_list[-2]
                    angle2_left, angle2_right = self.angle_list[-1]
                    angle_condition = 160 <= angle2_left <= 180
                    offset_condition = offset_left <= 80
                    if angle_condition and offset_condition:
                        A_color_left = (0, 255, 0)
                        self.info_messages.append("✅ Perfect Stance for Left!")
                        A_is_red_left = False
                        A_last_orange_time_left = 0.0
                    else:
                        if not angle_condition:
                            # print("angle2_left does not satisfy the condition.")
                            self.info_messages.append("⚠️ Keep Left Arm Straight(^v^)!")
                        if not offset_condition:
                            # print("offset_left does not satisfy the condition.")
                            self.info_messages.append("⚠️ Keep Left Forward(^v^)!")
                        if not A_is_red_left:
                            A_color_left = (0, 165, 255)
                            if A_last_orange_time_left == 0.0:
                                A_last_orange_time_left = time.time()
                            elif time.time() - A_last_orange_time_left >= 1.5:
                                A_color_left = (0, 0, 255)
                                A_is_red_left = True
                                if not angle_condition:
                                    self.count_angle_left += 1
                                if not offset_condition:
                                    self.count_offset_left += 1
                        else:
                            A_color_left = (0, 0, 255)
                            self.info_messages.extend(["🔴 Please Watch the Left Arm Stance!",
                                f"Count (Left Angle Error): {self.count_angle_left}",
                                f"Count (Left Offset Error): {self.count_offset_left}"])
                    angle_condition = 160 <= angle2_right <= 180
                    offset_condition = offset_right <= 80
                    if angle_condition and offset_condition:
                        A_color_right = (0, 255, 0)
                        self.info_messages.append("✅ Perfect Stance for Right!")
                        A_is_red_right = False
                        A_last_orange_time_right = 0.0
                    else:
                        if not angle_condition:
                            # print("angle2_right does not satisfy the condition.")
                            self.info_messages.append("⚠️ Keep Right Arm Straight(^v^)!")
                        if not offset_condition:
                            # print("offset_right does not satisfy the condition.")
                            self.info_messages.append("⚠️ Keep Right Forward(^v^)!")
                        if not A_is_red_right:
                            A_color_right = (0, 165, 255)  #
                            if A_last_orange_time_right == 0.0:
                                A_last_orange_time_right = time.time()
                            elif time.time() - A_last_orange_time_right >= 1.5:
                                A_color_right = (0, 0, 255)
                                A_is_red_right = True
                                if not angle_condition:
                                    self.count_angle_right += 1
                                if not offset_condition:
                                    self.count_offset_right += 1
                        else:
                            A_color_right = (0, 0, 255)
                            self.info_messages.extend([
                                "🔴 Please Watch the Right Arm Stance!",
                                f"Count (Right Angle Error): {self.count_angle_right}",
                                f"Count (Right Offset Error): {self.count_offset_right}"])
                    detector.Angle(v_pro, 11, 13, 15, True, color=A_color_left)
                    detector.Angle(v_pro, 12, 14, 16, True, color=A_color_right)
                    cv2.putText(v_pro, f"Offset 13-23: {offset_left}", (100, 200), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 0, 255), 5)
                    cv2.putText(v_pro, f"Offset 14-24: {offset_right}", (100, 250), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 0, 255), 5)
                draw_left_hip = detector.Angle(v_pro, 11, 23, 24, True, color=(255, 255, 255))
                draw_right_hip = detector.Angle(v_pro, 11, 12, 24, True, color=(255, 255, 255))
                cv2.putText(v_pro, f"Left: {angle_left}", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 5)
                cv2.putText(v_pro, f"Right: {angle_right}", (100, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 5)
                etime = time.time()
                fps = 1 / (etime - stime)
                stime = etime
                cv2.putText(v_pro, f"FPS: {int(fps)}", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 4)
            v_pro_rgb = cv2.cvtColor(v_pro, cv2.COLOR_BGR2RGB)
            image = QImage(v_pro_rgb.data, v_pro_rgb.shape[1], v_pro_rgb.shape[0], QImage.Format_RGB888)
            if window_name == "Video1":
                self.video1_display.set_image(image)
            elif window_name == "Video2":
                self.video2_display.set_image(image)
        callback()
    def update_ui(self):
        """更新进度条的值"""
        if self.total_frames_video1 > 0:
            progress = int((self.current_frame_video1 / self.total_frames_video1) * 100)
            self.progress_bar.setValue(progress)
        if self.info_messages:
            # 分类整理消息（新增绿色分类）
            green_left = [msg for msg in self.info_messages if "✅" in msg and "Left" in msg]
            green_right = [msg for msg in self.info_messages if "✅" in msg and "Right" in msg]
            orange_left = [msg for msg in self.info_messages if "⚠️" in msg and "Left" in msg] if not green_left else []
            orange_right = [msg for msg in self.info_messages if"⚠️" in msg and "Right" in msg] if not green_right else []
            red_left = [msg for msg in self.info_messages if "Left Arm Stance" in msg or "Count (Left" in msg]
            red_right = [msg for msg in self.info_messages if "Right Arm Stance" in msg or "Count (Right" in msg]

            # 修改HTML模板
            html_content = """
            <div style='font-size: 14pt;'>
                <table cellspacing="10">
                    <tr>
                        <td style='color: {color_left}; width: 10%;'>{left_msg}</td>
                    </tr>
                    <tr>
                        <td style='color: {color_right};'>{right_msg}</td>
                    </tr>
                    <tr>
                        <td style='color: red;'>{red_left}</td>
                    </tr>
                    <tr>
                        <td style='color: red;'>{red_right}</td>
                    </tr>
                </table>
            </div>
            """.format(
                color_left="green" if green_left else "orange",
                left_msg="&nbsp;&nbsp;|&nbsp;&nbsp;".join(green_left or orange_left),
                color_right="green" if green_right else "orange",
                right_msg="&nbsp;&nbsp;|&nbsp;&nbsp;".join(green_right or orange_right),
                red_left="&nbsp;&nbsp;|&nbsp;&nbsp;".join(red_left),
                red_right="&nbsp;&nbsp;|&nbsp;&nbsp;".join(red_right)
            )

            self.text_browser_videodata.setHtml(html_content)
            self.info_messages.clear()
    def update_playback_lcd(self):
        if self.enable_playback_radio.isChecked():
            self.playback_time += 1
            self.playback_lcd.display(self.playback_time)

    # ======================================================
    def update_table_data(self):
        train_count = 0
        if hasattr(self, 'video_files') and len(self.video_files) > 0:
            train_count = len(self.video_files)
        elif self.key_edit1.text().strip():
            train_count = 1

        playback_path = self.playback_edit.text() if self.playback_edit.text() != self.playback_edit.placeholderText() else "None"
        report_path = self.report_edit.text() if self.report_edit.text() != self.report_edit.placeholderText() else "None"
        self.total_errors = (self.count_angle_left + self.count_offset_left +
                            self.count_angle_right + self.count_offset_right)
        ANGLE_THRESHOLD = 8
        OFFSET_THRESHOLD = 5
        base_score = 10.0
        angle_penalty = (self.count_angle_left + self.count_angle_right) * 0.2
        offset_penalty = (self.count_offset_left + self.count_offset_right) * 0.1
        overall_score = max(0, base_score - angle_penalty - offset_penalty)
        if overall_score >= 9:
            score_grade = "优秀"
        elif overall_score >= 7:
            score_grade = "良好"
        elif overall_score >= 5:
            score_grade = "一般"
        else:
            score_grade = "需要改进"
        guidance_recommendations = []
        if self.count_angle_left < ANGLE_THRESHOLD:
            guidance_recommendations.append("✅ 左臂角度控制得很好，请继续保持！")
        else:
            guidance_recommendations.append("🔴 请调整左臂的角度，确保动作更加标准。")
        if self.count_offset_left < OFFSET_THRESHOLD:
            guidance_recommendations.append("✅ 左臂位置控制得很好，请继续保持！")
        else:
            guidance_recommendations.append("🔴 请注意左臂的位置，避免过度偏离中心线。")
        if self.count_angle_right < ANGLE_THRESHOLD:
            guidance_recommendations.append("✅ 右臂角度控制得很好，请继续保持！")
        else:
            guidance_recommendations.append("🔴 请调整右臂的角度，确保动作更加标准。")
        if self.count_offset_right < OFFSET_THRESHOLD:
            guidance_recommendations.append("✅ 右臂位置控制得很好，请继续保持！")
        else:
            guidance_recommendations.append("🔴 请注意右臂的位置，避免过度偏离中心线。")
        if self.total_errors == 0:
            guidance_recommendations.append("🎉 整体表现非常出色，请继续保持！")
        elif self.total_errors < 5:
            guidance_recommendations.append("😊 整体表现不错，但仍有提升空间，请继续努力！")
        else:
            guidance_recommendations.append("⚠️ 整体表现需要改进，请重点关注错误较多的动作。")
        guidance_text = "\n".join(guidance_recommendations)
        row = [
            self.start_time, self.end_time,
            self.demo_video_path, playback_path, report_path,
            str(train_count), str(self.total_errors),
            str(self.count_angle_left), str(self.count_offset_left),
            str(self.count_angle_right), str(self.count_offset_right),
            guidance_text,f"{overall_score:.1f} ({score_grade})"]

        self.table_model.appendRow([QStandardItem(item) for item in row])
    # ===========================================================================

    def table_data_off(self):
        report_path = "'Report'-OFF"
        row = [
            self.start_time, self.end_time,
            report_path]
        self.table_model.appendRow([QStandardItem(item) for item in row])
    def closeEvent(self, event):
        self.stop_detection()
        event.accept()