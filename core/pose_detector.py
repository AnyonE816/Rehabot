# core/pose_detector.py
import cv2
import math
import numpy as np
import mediapipe as mp


class PoseDetector:
    """
    实时姿态检测核心类
    基于MediaPipe实现人体关键点检测、角度计算、偏移量计算
    """

    def __init__(self, mode=False, complexity=1, smooth=True,
                 en_segmentation=False, sm_segment=True, detection=0.5, tracking=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.en_segmentation = en_segmentation
        self.sm_segment = sm_segment
        self.detection = detection
        self.tracking = tracking

        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(
            self.mode, self.complexity, self.smooth,
            self.en_segmentation, self.sm_segment, self.detection, self.tracking
        )
        self.lmlist = []

    def find_position(self, video, draw=False):
        """检测人体关键点坐标"""
        videoRGB = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)
        results = self.pose.process(videoRGB)

        self.lmlist = []
        if results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(video, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = video.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(video, (cx, cy), 6, (255, 105, 180), cv2.FILLED)

        return video, self.lmlist

    def calculate_angle(self, video, p1, p2, p3, draw=True, color=(255, 255, 255)):
        """计算三个关键点之间的夹角"""
        x1, y1 = self.lmlist[p1][1:]
        x2, y2 = self.lmlist[p2][1:]
        x3, y3 = self.lmlist[p3][1:]

        # 向量计算
        v1 = (x1 - x2, y1 - y2)
        v2 = (x3 - x2, y3 - y2)
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        magnitude_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        magnitude_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
        cos_theta = max(-1, min(1, cos_theta))
        angle = round(math.degrees(math.acos(cos_theta)), 1)

        if draw:
            cv2.line(video, (x1, y1), (x2, y2), color, 5)
            cv2.line(video, (x2, y2), (x3, y3), color, 5)
            circle_color = (255, 105, 180)
            cv2.circle(video, (x1, y1), 10, circle_color, cv2.FILLED)
            cv2.circle(video, (x2, y2), 10, circle_color, cv2.FILLED)
            cv2.circle(video, (x3, y3), 10, circle_color, cv2.FILLED)

        return angle

    def calculate_x_offset(self, p1, p2):
        """计算两个关键点的X轴偏移量"""
        x1, _ = self.lmlist[p1][1:]
        x2, _ = self.lmlist[p2][1:]
        return abs(x1 - x2)